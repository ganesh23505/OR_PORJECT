# ap.py
# Chennai evacuation API (transportation + assignment) using chennai_network.json
# Run:  pip install -r requirements.txt
#       python app.py

import json
import math
import random
import time
import os
from typing import Dict, List, Tuple

import networkx as nx
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ------------- Config -------------
NETWORK_JSON_PATH = os.path.join(os.path.dirname(__file__), "chennai_network.json")
DEFAULT_SPEED_KMPH = 30.0
VEHICLE_DISPATCH_PREP_MINS = 10
# ----------------------------------

app = Flask(__name__)
CORS(app)

# --- Load the network at startup (for Render / gunicorn) ---
graph = None
try:
    graph = None
    print(f"Attempting to load network from: {NETWORK_JSON_PATH}")
    graph = None
    import os
    from pathlib import Path
    if not Path(NETWORK_JSON_PATH).exists():
        print(f"❌ File not found at {NETWORK_JSON_PATH}")
    else:
        from ap import load_network, set_cost_mode  # if your function is defined below
    # But in your file it’s already defined before usage — so just call directly:
    from __main__ import load_network, set_cost_mode  # remove this if not needed
except Exception:
    pass

# For now, just call load_network directly (since it’s defined later)
try:
    from __main__ import load_network, set_cost_mode
except ImportError:
    pass

# --- Simpler way ---
try:
    graph = load_network(NETWORK_JSON_PATH)
    set_cost_mode("time")
    print("✅ Network successfully loaded at startup.")
except Exception as e:
    print(f"❌ Failed to load network at startup: {e}")
    graph = None
# ----------------------------------------------------------

    
# --- Globals populated at startup ---
graph: nx.DiGraph = None
nodes_data: Dict[str, dict] = {}
shelter_ids: List[str] = []
zone_ids: List[str] = []
shelter_names: Dict[str, str] = {}
server_start_time = time.time() # To simulate time

# --- NEW: Server-side state for shelter occupancy and vehicles ---
# These are populated by load_network()
shelter_capacities: Dict[str, int] = {}
shelter_occupancy: Dict[str, int] = {}
vehicle_fleet: List[Dict] = []
# ----------------------------------------------------------------

def get_current_time_minutes():
    """Returns server uptime in minutes for simulation."""
    return (time.time() - server_start_time) / 60.0

# ===================== Load Network & Init State =====================

def load_network(json_path: str) -> nx.DiGraph:
    """
    Build a directed graph from chennai_network.json.
    Initializes shelter capacities, occupancy, and vehicle fleet.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    arcs = data["arcs"]
    G = nx.DiGraph()

    # Add nodes (name + coordinates)
    for nd in nodes:
        nid = nd["id"]
        G.add_node(
            nid,
            id=nid,
            name=nd.get("name", nid),
            lat=float(nd.get("lat", 0.0)),
            lon=float(nd.get("lon", 0.0)),
            kind=("shelter" if nid.startswith("s") else "zone")
        )

    # Add directed arcs
    speed_km_per_min = DEFAULT_SPEED_KMPH / 60.0
    for e in arcs:
        u, v = e["from"], e["to"]
        dist_km = float(e.get("distance", 0.0))
        time_min = max(0.5, dist_km / max(1e-9, speed_km_per_min))
        G.add_edge(u, v,
                   distance_km=dist_km,
                   travel_time_min=time_min,
                   weight=time_min)

    # Add reverse edges if missing (symmetrize)
    added_back = 0
    for u, v, d in list(G.edges(data=True)):
        if not G.has_edge(v, u):
            G.add_edge(v, u, **d)
            added_back += 1

    # Fill globals for convenience
    global nodes_data, shelter_ids, zone_ids, shelter_names
    nodes_data = {n: G.nodes[n] for n in G.nodes()}
    shelter_ids = sorted([n for n in G.nodes() if str(n).startswith("s")])
    zone_ids = sorted([n for n in G.nodes() if str(n).startswith("n")])
    shelter_names = {sid: nodes_data[sid].get("name", sid) for sid in shelter_ids}

    # --- NEW: Initialize Server State ---
    global shelter_capacities, shelter_occupancy, vehicle_fleet
    
    # 1. Init Shelter Capacities & Occupancy
    for sid in shelter_ids:
        shelter_capacities[sid] = 5000
        shelter_occupancy[sid] = random.randint(1000, 4500) # Randomly fill
    
    # Make one shelter almost full
    if shelter_ids:
        # Use a shelter ID that is guaranteed to exist
        test_shelter_id = shelter_ids[0] # s1 (Guindy)
        shelter_occupancy[test_shelter_id] = 4980 
        print(f"--- TEST STATE: Shelter {test_shelter_id} ({shelter_names.get(test_shelter_id)}) set to 4980 occupancy ---")
    
    # 2. Init Vehicle Fleet (5 per shelter)
    vehicle_fleet = []
    current_time_min = get_current_time_minutes()
    for sid in shelter_ids:
        for i in range(5):
            status = random.choice(["available", "available", "engaged"])
            available_at = current_time_min # Available now
            if status == "engaged":
                # Becomes free in 5-60 mins from now
                available_at = current_time_min + random.randint(5, 60) 
                
            vehicle_fleet.append({
                "id": f"v_{sid}_{i+1}",
                "baseShelterId": sid,
                "capacity": random.choice([10, 20, 50]), # Vehicle size
                "status": status,
                "available_at": available_at # Time in (simulated) minutes
            })
    # --- END NEW STATE ---

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges")
    print(f"Zones: {len(zone_ids)}, Shelters: {len(shelter_ids)}")
    print("--- Server State Initialized ---")
    if shelter_ids:
        print(f"Shelter 1 Occupancy: {shelter_occupancy.get(shelter_ids[0])} / {shelter_capacities.get(shelter_ids[0])}")
    print(f"Total Vehicles: {len(vehicle_fleet)}")
    print(f"Available Vehicles: {len([v for v in vehicle_fleet if v['status'] == 'available'])}")
    print("---------------------------------")
    
    return G


# ===================== Helpers =====================

def set_cost_mode(mode: str):
    if mode not in {"time", "distance"}:
        mode = "time"
    key = "travel_time_min" if mode == "time" else "distance_km"
    for u, v, d in graph.edges(data=True):
        d["weight"] = float(d.get(key, 1.0))

def weighted_cost(u: str, v: str) -> float:
    """Helper to get weighted cost, returns infinity on no path."""
    try:
        return float(nx.shortest_path_length(graph, source=u, target=v, weight="weight"))
    except nx.NetworkXNoPath:
        return float("inf")
    except nx.NodeNotFound:
        return float("inf")

def shortest_path_and_dist(u: str, v: str) -> Tuple[float, List[str]]:
    """
    Return (weighted_distance, path_nodes).
    Raises nx.NetworkXNoPath if no path.
    """
    dist = weighted_cost(u, v)
    if math.isinf(dist):
        raise nx.NetworkXNoPath
    path = nx.shortest_path(graph, source=u, target=v, weight="weight")
    return float(dist), path

def shelters_sorted_by_distance_from(start_node_id: str) -> List[Tuple[str, float, List[str]]]:
    """
    List shelters sorted by (start -> shelter) cost using current edge 'weight'.
    """
    cand = []
    for s in shelter_ids:
        try:
            d, p = shortest_path_and_dist(start_node_id, s)
            cand.append((s, d, p))
        except nx.NetworkXNoPath:
            continue
    cand.sort(key=lambda x: x[1])
    return cand


# ===================== API =====================

@app.route("/api/get-locations", methods=["GET"])
def get_locations():
    if not zone_ids:
        return jsonify({"error": "Network data not loaded"}), 500
    locations_list = sorted(
        [nodes_data[zid] for zid in zone_ids], 
        key=lambda x: x['name']
    )
    return jsonify(locations_list)

@app.route("/api/get-network-data", methods=["GET"])
def get_network_data():
    try:
        with open(NETWORK_JSON_PATH, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "Could not read network file"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/api/meta", methods=["GET"])
def meta():
    # Return current state
    current_time_min = get_current_time_minutes()
    # Check for vehicles that *were* engaged but are now free
    available_count = 0
    for v in vehicle_fleet:
        if v["status"] == "available":
            available_count += 1
        elif v["status"] == "engaged" and current_time_min >= v["available_at"]:
            v["status"] = "available" # Make it available again
            available_count += 1

    return jsonify({
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "zones": len(zone_ids),
        "shelters": len(shelter_ids),
        "shelter_occupancy": shelter_occupancy,
        "shelter_capacities": shelter_capacities,
        "vehicles_total": len(vehicle_fleet),
        "vehicles_available": available_count
    })

@app.route("/api/shelters", methods=["GET"])
def list_shelters():
    out = []
    for sid in shelter_ids:
        nd = nodes_data.get(sid, {})
        out.append({
            "id": sid,
            "name": nd.get("name", sid),
            "lat": nd.get("lat"),
            "lon": nd.get("lon"),
            # NEW: Add live capacity
            "capacity": shelter_capacities.get(sid, 0),
            "occupancy": shelter_occupancy.get(sid, 0),
            "remaining": shelter_capacities.get(sid, 0) - shelter_occupancy.get(sid, 0)
        })
    return jsonify(out)


# ---------- Transportation: capacity-aware allocation ----------

@app.route("/api/transport-allocate", methods=["POST"])
def transport_allocate():
    """
    MODIFIED: This function now uses the *global server state* for shelter capacity
    and updates it when an allocation is made.
    """
    if graph is None or not shelter_ids:
        return jsonify({"error": "Network not loaded"}), 500

    body = request.get_json(force=True)
    groups = body.get("groups", [])
    allow_partial = bool(body.get("allowPartialSplit", False))
    cost_mode = body.get("costMode", "time")
    set_cost_mode(cost_mode)

    global shelter_occupancy

    allocations = []
    unassigned = []

    for g in groups:
        start = g.get("startNodeId")
        people = int(g.get("people", 0))
        if not start or start not in nodes_data or people <= 0:
            continue

        candidates = shelters_sorted_by_distance_from(start)
        placed = False
        
        if not allow_partial:
            for (sid, cost_time, path_ids) in candidates:
                current_occ = shelter_occupancy.get(sid, 0)
                max_cap = shelter_capacities.get(sid, 0)
                remaining_cap = max_cap - current_occ
                
                if remaining_cap >= people:
                    shelter_occupancy[sid] += people 
                    
                    dist_km = 0
                    for i in range(len(path_ids) - 1):
                        dist_km += graph[path_ids[i]][path_ids[i+1]].get('distance_km', 0)
                        
                    allocations.append({
                        "startNodeId": start,
                        "people": people,
                        "assignedShelterId": sid,
                        "assignedShelterName": shelter_names.get(sid, sid),
                        "cost_time_min": cost_time,
                        "distance_km": dist_km,
                        "path_ids": path_ids,
                        "path": [nodes_data[n]["name"] for n in path_ids]
                    })
                    placed = True
                    print(f"Allocation: {people} from {start} to {sid}. New occupancy: {shelter_occupancy[sid]}")
                    break
        else:
            # partial split logic
            rem = people
            partials = []
            for (sid, cost_time, path_ids) in candidates:
                if rem <= 0: break
                
                current_occ = shelter_occupancy.get(sid, 0)
                max_cap = shelter_capacities.get(sid, 0)
                remaining_cap = max_cap - current_occ
                
                if remaining_cap <= 0:
                    continue
                
                take = min(remaining_cap, rem)
                shelter_occupancy[sid] += take
                rem -= take
                
                dist_km = 0
                for i in range(len(path_ids) - 1):
                    dist_km += graph[path_ids[i]][path_ids[i+1]].get('distance_km', 0)
                
                partials.append({
                    "startNodeId": start,
                    "people": int(take),
                    "assignedShelterId": sid,
                    "assignedShelterName": shelter_names.get(sid, sid),
                    "cost_time_min": cost_time,
                    "distance_km": dist_km,
                    "path_ids": path_ids,
                    "path": [nodes_data[n]["name"] for n in path_ids]
                })
                print(f"Partial Allocation: {take} from {start} to {sid}. New occupancy: {shelter_occupancy[sid]}")

            if partials and rem == 0:
                allocations.extend(partials)
                placed = True

        if not placed:
            unassigned.append({"startNodeId": start, "people": people})
            print(f"Unassigned: {people} from {start}. No shelters with capacity.")


    return jsonify({
        "allocations": allocations,
        "remainingShelterCapacity": {sid: shelter_capacities.get(sid, 5000) - shelter_occupancy.get(sid, 0) for sid in shelter_ids},
        "unassigned": unassigned
    })


# --- MODIFIED: Transport Request API with ETA ---
@app.route("/api/request-transport", methods=["POST"])
def request_transport():
    """
    MODIFIED: Checks for available vehicles and provides a real ETA.
    If vehicles are engaged, it calculates ETA based on when they become free.
    """
    global vehicle_fleet
    
    body = request.get_json(force=True)
    shelter_id = body.get("shelterId")
    start_node_id = body.get("startNodeId") # NEW: Need user's location
    people_count = int(body.get("people", 0))
    
    if not shelter_id or not start_node_id or people_count <= 0:
        return jsonify({"status": "error", "message": "Invalid request."}), 400
    
    if shelter_id not in shelter_names or start_node_id not in nodes_data:
        return jsonify({"status": "error", "message": "Invalid location or shelter ID."}), 400

    # Calculate travel time from shelter to user's start location
    set_cost_mode("time") # Ensure we are using time
    travel_time_to_user = weighted_cost(shelter_id, start_node_id)
    
    if math.isinf(travel_time_to_user):
        return jsonify({"status": "error", "message": "Cannot calculate route from shelter to your location."})

    current_time_min = get_current_time_minutes()
    
    # Check for vehicles that *were* engaged but are now free
    for v in vehicle_fleet:
        if v["status"] == "engaged" and current_time_min >= v["available_at"]:
            v["status"] = "available"
            v["available_at"] = current_time_min
            print(f"Vehicle {v['id']} is now available.")

    # Find available vehicles *at that shelter*
    available_vehicles = [
        v for v in vehicle_fleet 
        if v["baseShelterId"] == shelter_id and v["status"] == "available"
    ]
    
    # 1. Try to find a single, available vehicle that can take the whole group
    capable_vehicles = sorted(
        [v for v in available_vehicles if v["capacity"] >= people_count],
        key=lambda x: x["capacity"] # Find smallest capable vehicle
    )
    
    if capable_vehicles:
        vehicle_to_assign = capable_vehicles[0]
        # --- UPDATE STATE ---
        vehicle_to_assign["status"] = "engaged"
        # Calculate when it will be free again (round trip + 10 min buffer)
        trip_duration = (travel_time_to_user * 2) + VEHICLE_DISPATCH_PREP_MINS
        vehicle_to_assign["available_at"] = current_time_min + trip_duration
        # ---
        
        total_eta_mins = travel_time_to_user + VEHICLE_DISPATCH_PREP_MINS
        
        print(f"Vehicle {vehicle_to_assign['id']} assigned. ETA: {total_eta_mins:.0f} mins.")
        return jsonify({
            "status": "assigned",
            "vehicleId": vehicle_to_assign["id"],
            "message": f"Vehicle {vehicle_to_assign['id']} (Capacity: {vehicle_to_assign['capacity']}) dispatched. Arrives in approx. {total_eta_mins:.0f} minutes."
        })

    # 2. If no *available* vehicle, check *engaged* vehicles
    engaged_vehicles = sorted(
        [v for v in vehicle_fleet if v["baseShelterId"] == shelter_id and v["capacity"] >= people_count],
        key=lambda x: x["available_at"] # Find one that is free the soonest
    )
    
    if engaged_vehicles:
        next_vehicle = engaged_vehicles[0]
        wait_time_mins = next_vehicle["available_at"] - current_time_min
        total_eta_mins = wait_time_mins + travel_time_to_user + VEHICLE_DISPATCH_PREP_MINS
        
        print(f"All vehicles busy. Next available: {next_vehicle['id']} in {wait_time_mins:.0f} mins. Total ETA: {total_eta_mins:.0f} mins.")
        return jsonify({
            "status": "pending",
            "vehicleId": next_vehicle["id"],
            "message": f"All vehicles are busy. The next available vehicle ({next_vehicle['id']}) will be free in {wait_time_mins:.0f} mins. Total ETA: approx. {total_eta_mins:.0f} minutes."
        })

    # 3. If not enough capacity at all (no vehicle is big enough)
    print(f"No vehicle capacity for {people_count} at {shelter_id}.")
    return jsonify({
        "status": "error",
        "message": f"No vehicles with sufficient capacity ({people_count} people) are based at {shelter_names[shelter_id]}. Please try to arrange your own transport."
    })


# ===================== Main =====================
from flask import send_from_directory

@app.route('/')
def serve_index():
    # Serve the main page (index.html)
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    # Serve other static files (JS, CSS, etc.)
    return send_from_directory('.', path)

if __name__ == "__main__":
    graph = load_network(NETWORK_JSON_PATH)
    # default cost mode = time (minutes)
    set_cost_mode("time")
    app.run(host="0.0.0.0", port=5000, debug=True)




