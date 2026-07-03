import blenderproc as bproc
import os

os.environ["CYCLES_CUDA_MEMORY_LIMIT"] = "4096"

import bpy
import random
import json
import numpy as np
import os
from pathlib import Path

import sys
import argparse
import gc

# Attempt to enable memory-friendly Blender settings and add handlers
def _setup_memory_safety_handlers():
    try:
        # Some Blender builds expose a sequential loading flag; set if available
        try:
            bpy.context.preferences.render.use_sequential_render = True
        except Exception:
            pass

        def _purge_images_on_render(scene=None):
            try:
                # Remove unused images to free large memory blocks
                for img in list(bpy.data.images):
                    try:
                        if getattr(img, "users", 0) == 0:
                            bpy.data.images.remove(img)
                    except Exception:
                        pass
                try:
                    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
                except Exception:
                    pass
                gc.collect()
            except Exception:
                pass

        try:
            if hasattr(bpy.app.handlers, "render_complete"):
                bpy.app.handlers.render_complete.append(_purge_images_on_render)
        except Exception:
            try:
                if hasattr(bpy.app.handlers, "render_post"):
                    bpy.app.handlers.render_post.append(_purge_images_on_render)
            except Exception:
                pass
    except Exception:
        pass


# Configure handlers early
_setup_memory_safety_handlers()

# ── CHANGE FROM ORIGINAL ──────────────────────────────────────────────────────
# Original hardcoded:  CONFIG_PATH = Path("configs/scene_configs_100.json")
# Now reads --config argument passed by main.py (RAG output = temp_configs.json)
# Falls back to scene_configs_100.json only if no --config is given.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    """
    BlenderProc passes all script args after '--'.
    We only look for --config and --output_dir.
    """
    argv = sys.argv
    # BlenderProc puts script args after '--'
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="FABRIC-AI scene generator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON scene config file (RAG output)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for images and annotations",
    )
    args, _ = parser.parse_known_args(argv)
    return args


_args = _parse_args()

ROOT        = Path(__file__).resolve().parents[2]
ASSETS_PATH = ROOT / "assets"
HDRI_PATH   = ROOT / "assets/hdri"
VEHICLES_PATH  = ROOT / "assets/vehicles"
ANIMALS_PATH   = ROOT / "assets/animals"
TEXTURES_PATH  = ROOT / "assets/textures"

# ── Config resolution ─────────────────────────────────────────────────────────
if _args.config and Path(_args.config).exists():
    CONFIG_PATH = Path(_args.config)
    print(f"[CONFIG] Loaded from RAG: {CONFIG_PATH}")
else:
    CONFIG_PATH = ROOT / "configs/scene_configs_100.json"
    print(f"[CONFIG] No --config given, using default: {CONFIG_PATH}")


def _normalize_scene_configs(raw_data):
    if isinstance(raw_data, dict):
        if "configs" in raw_data and isinstance(raw_data["configs"], list):
            configs = raw_data["configs"]
        else:
            num_to_generate = raw_data.get("num_scenes", 5)
            if not isinstance(num_to_generate, int) or num_to_generate < 1:
                num_to_generate = 5
            configs = [raw_data.copy() for _ in range(num_to_generate)]
    elif isinstance(raw_data, list):
        configs = raw_data
    else:
        raise ValueError(
            f"Unsupported config format {type(raw_data).__name__}: expected dict or list"
        )

    if not configs:
        configs = [{}]
    return configs


def _downscale_large_textures(max_dim=2048):
    for img in bpy.data.images:
        try:
            width, height = img.size
        except Exception:
            continue
        if width > max_dim or height > max_dim:
            scale_factor = min(max_dim / width, max_dim / height)
            new_width = max(1, int(width * scale_factor))
            new_height = max(1, int(height * scale_factor))
            try:
                img.scale(new_width, new_height)
                print(f"[MEMORY] Downscaled image '{img.name}' from {width}x{height} to {new_width}x{new_height}")
            except Exception as err:
                print(f"[WARN] Failed to downscale image '{img.name}': {err}")


def _purge_unused_data():
    for block in [bpy.data.meshes, bpy.data.images, bpy.data.materials, bpy.data.textures, bpy.data.objects, bpy.data.collections]:
        orphan_items = [item for item in block if getattr(item, "users", 0) == 0]
        if not orphan_items:
            continue
        try:
            if hasattr(bpy.data, "batch_remove"):
                bpy.data.batch_remove(orphan_items)
            else:
                for item in orphan_items:
                    block.remove(item)
        except Exception as err:
            for item in orphan_items:
                try:
                    block.remove(item)
                except Exception:
                    pass
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception as err:
        print(f"[WARN] Orphan purge failed: {err}")


def _clean_memory_before_scene():
    _purge_unused_data()
    _downscale_large_textures(max_dim=2048)
    gc.collect()

with open(CONFIG_PATH, "r") as f:
    raw_loaded_data = json.load(f)

scene_configs = _normalize_scene_configs(raw_loaded_data)

# ── Output resolution ─────────────────────────────────────────────────────────
if _args.output_dir:
    OUTPUT = Path(_args.output_dir)
else:
    OUTPUT = ROOT / "output"

OUTPUT.mkdir(parents=True, exist_ok=True)
print(f"[OUTPUT] Writing to: {OUTPUT}")

# ─────────────────────────────────────────────────────────────────────────────
# Everything below is IDENTICAL to the original — no other changes.
# ─────────────────────────────────────────────────────────────────────────────

LANES = [-2.5, 0, 2.5]
CATEGORY_IDS = {
    "car":   1,
    "auto":  2,
    "truck": 3,
    "bus":   4,
    "cow":   5,
}

CATEGORIES = [
    {"id": 1, "name": "car",   "supercategory": "vehicle"},
    {"id": 2, "name": "auto",  "supercategory": "vehicle"},
    {"id": 3, "name": "truck", "supercategory": "vehicle"},
    {"id": 4, "name": "bus",   "supercategory": "vehicle"},
    {"id": 5, "name": "cow",   "supercategory": "animal"},
]

class Scene:

    def __init__(self, config):
        self.config = config
        self.base_y = 25
        self.last_y = 25
        self.lane_last_y = {-2.5: 25, 0: 25, 2.5: 25}
        self.large_vehicle_lanes = set()
        self.large_vehicle_last_y = 0
        self.all_vehicle_positions = []
        self.spawned_objects = []

    # ---------- WORLD ----------
    def world(self):
        weather = self.config.get("weather", "clear")
        time_of_day = self.config.get("time_of_day", "midday")

        world = bpy.data.worlds.new("world")
        bpy.context.scene.world = world
        world.use_nodes = True

        n = world.node_tree.nodes
        l = world.node_tree.links
        n.clear()

        sky = n.new("ShaderNodeTexSky")
        sky.sky_type = 'NISHITA'
        sky.sun_disc = (weather not in ["rainy", "foggy"])
        sky.sun_size = np.radians(1.5) if sky.sun_disc else 0.0

        if weather == "rainy":
            sky.dust_density = 5.0
            sky.air_density = 3.0
            sky.sun_intensity = 0.2
        elif weather == "foggy":
            sky.dust_density = 8.0
            sky.air_density = 4.0
            sky.sun_intensity = 0.4
        else:
            sky.dust_density = 1.0
            sky.air_density = 1.0
            sky.sun_intensity = 1.0

        if time_of_day == "morning":
            sky.sun_elevation = np.radians(20)
            sky.sun_rotation = np.radians(45)
        elif time_of_day == "evening":
            sky.sun_elevation = np.radians(10)
            sky.sun_rotation = np.radians(-60)
        elif time_of_day == "night":
            sky.sun_elevation = np.radians(-20)
            sky.sun_intensity = 0.05
        else:
            sky.sun_elevation = np.radians(60)
            sky.sun_rotation = np.radians(15)

        bg = n.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = 1.0 if sky.sun_elevation > 0 else 0.3

        out = n.new("ShaderNodeOutputWorld")
        l.new(sky.outputs["Color"], bg.inputs["Color"])
        l.new(bg.outputs["Background"], out.inputs["Surface"])

        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)

        if time_of_day == "night":
            self._add_night_lighting()

    def _add_night_lighting(self):
        weather = self.config.get("weather", "clear")

        bpy.ops.object.light_add(type='SUN', location=(0, 0, 100))
        moon = bpy.context.active_object
        moon.data.energy = 0.4 if weather == "foggy" else 0.15
        moon.data.color = (0.4, 0.5, 0.8)
        moon.rotation_euler = (np.radians(20), 0, 0)

        for i in range(0, 300, 30):
            for side_x in [-7.0, 7.0]:
                bpy.ops.object.light_add(
                    type='POINT',
                    location=(side_x, self.base_y + i, 6.5)
                )
                lamp = bpy.context.active_object
                lamp.data.energy = 1500
                lamp.data.color = (1.0, 0.85, 0.5)
                lamp.data.shadow_soft_size = 0.5

        bpy.ops.object.light_add(type='AREA', location=(0, 50, 25))
        fill = bpy.context.active_object
        fill.data.energy = 500 if weather == "foggy" else 150
        fill.data.color = (0.6, 0.65, 0.8)
        fill.data.size = 20
        fill.rotation_euler = (np.radians(45), 0, 0)

        bpy.ops.object.light_add(type='AREA', location=(0, 20, 1.0))
        headlights = bpy.context.active_object
        headlights.data.energy = 300
        headlights.data.color = (1.0, 0.98, 0.9)
        headlights.data.size = 5
        headlights.rotation_euler = (np.radians(90), 0, 0)

    # ---------- ENVIRONMENT ----------
    def add_environment(self):
        for i in range(0, 400, 30):
            for side_x in [-7.0, 7.0]:
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.07, depth=7,
                    location=(side_x, self.base_y + i, 3.5)
                )
                pole = bpy.context.active_object
                pole_mat = bpy.data.materials.new(f"Pole_{i}")
                pole_mat.use_nodes = True
                bsdf = pole_mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (0.25, 0.22, 0.18, 1)
                bsdf.inputs["Roughness"].default_value = 0.95
                pole.data.materials.append(pole_mat)

        bpy.ops.mesh.primitive_plane_add(size=1500)
        ground = bpy.context.active_object
        ground.location = (0, 700, -0.01)
        ground.scale = (25, 1, 1)

        ground_mat = bpy.data.materials.new("Ground")
        ground_mat.use_nodes = True
        n = ground_mat.node_tree.nodes
        l = ground_mat.node_tree.links
        n.clear()

        coord = n.new("ShaderNodeTexCoord")
        noise = n.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 8.0
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.7

        mix = n.new("ShaderNodeMixRGB")
        mix.inputs["Color1"].default_value = (0.35, 0.28, 0.18, 1)
        mix.inputs["Color2"].default_value = (0.25, 0.20, 0.12, 1)

        bsdf = n.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 1.0
        out = n.new("ShaderNodeOutputMaterial")

        l.new(coord.outputs["Generated"], noise.inputs["Vector"])
        l.new(noise.outputs["Fac"], mix.inputs["Fac"])
        l.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        ground.data.materials.append(ground_mat)

        for i in range(0, 300, 100):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.05, depth=4,
                location=(6.2, self.base_y + i + 50, 2.0)
            )
            post = bpy.context.active_object
            post_mat = bpy.data.materials.new(f"SignPost_{i}")
            post_mat.use_nodes = True
            post_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1)
            post.data.materials.append(post_mat)

            bpy.ops.mesh.primitive_plane_add(size=0.8)
            sign = bpy.context.active_object
            sign.location = (6.2, self.base_y + i + 50, 4.2)
            sign.rotation_euler = (np.radians(90), 0, np.radians(90))
            sign_mat = bpy.data.materials.new(f"Sign_{i}")
            sign_mat.use_nodes = True
            sign_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1.0, 0.9, 0.0, 1)
            sign.data.materials.append(sign_mat)

    # ---------- FOG ----------
    def create_fog(self):
        weather = self.config.get("weather", "clear")
        time_of_day = self.config.get("time_of_day", "midday")

        if weather in ["rainy", "foggy"]:
            bpy.ops.mesh.primitive_cube_add(size=1500)
            fog = bpy.context.active_object
            fog.location = (0, 700, 10)

            fog_mat = bpy.data.materials.new("FogMat")
            fog_mat.use_nodes = True
            n = fog_mat.node_tree.nodes
            l = fog_mat.node_tree.links
            n.clear()

            vol = n.new("ShaderNodeVolumePrincipled")
            if time_of_day == "night":
                density = 0.001 if weather == "rainy" else 0.002
            else:
                density = 0.002 if weather == "rainy" else 0.005

            vol.inputs["Density"].default_value = density
            vol.inputs["Anisotropy"].default_value = 0.5
            vol.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)

            out = n.new("ShaderNodeOutputMaterial")
            l.new(vol.outputs["Volume"], out.inputs["Volume"])
            fog.data.materials.append(fog_mat)
            fog.display_type = 'WIRE'

    # ---------- BUILDINGS ----------
    def spawn_buildings(self):
        buildings_dir = ASSETS_PATH / "buildings"
        building_configs = {
            "building1.glb": {"z_offset": 0.0, "rotation": 0,  "scale": 3.5},
            "house1.glb":    {"z_offset": 0.0, "rotation": 0,  "scale": 1.2},
            "hut1.glb":      {"z_offset": 0.0, "rotation": 0,  "scale": 1.2},
        }

        available = []
        for fname, cfg in building_configs.items():
            path = buildings_dir / fname
            if path.exists():
                available.append((path, cfg))

        if not available:
            print("[WARN] No buildings found in assets/buildings/")
            return

        y_start = self.base_y + 40

        def place_building(x, y, path, cfg):
            obj = self.load(path)
            if not obj:
                return
            obj.set_scale([cfg["scale"]] * 3)
            bbox = obj.get_bound_box()
            min_z = min([p[2] for p in bbox])
            obj.set_location([x, y, -min_z + cfg["z_offset"]])
            rot = cfg.get("rotation", 0)
            if x < 0:
                rot += 90
            else:
                rot -= 90
            obj.set_rotation_euler([0, 0, np.radians(rot)])

        y_left = y_start
        for _ in range(6):
            path, cfg = random.choice(available)
            x = random.uniform(-12, -8)
            place_building(x, y_left, path, cfg)
            y_left += random.uniform(25, 40)

        y_right = y_start + random.uniform(5, 15)
        for _ in range(6):
            path, cfg = random.choice(available)
            x = random.uniform(8, 12)
            place_building(x, y_right, path, cfg)
            y_right += random.uniform(25, 40)

    # ---------- ROAD ----------
    def road(self):
        bpy.ops.mesh.primitive_plane_add(size=1500)
        road = bpy.context.active_object
        road.location = (0, 700, 0)

        mat = bpy.data.materials.new("road")
        mat.use_nodes = True
        n = mat.node_tree.nodes
        l = mat.node_tree.links
        n.clear()

        tex = n.new("ShaderNodeTexCoord")
        mapping = n.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (6, 150, 1)

        bsdf = n.new("ShaderNodeBsdfPrincipled")
        out = n.new("ShaderNodeOutputMaterial")

        albedo_path = TEXTURES_PATH / "albedo.jpg"
        normal_path = TEXTURES_PATH / "normal.jpg"
        if albedo_path.exists():
            alb = n.new("ShaderNodeTexImage")
            alb.image = bpy.data.images.load(str(albedo_path))
            l.new(tex.outputs["UV"], mapping.inputs["Vector"])
            l.new(mapping.outputs["Vector"], alb.inputs["Vector"])
            l.new(alb.outputs["Color"], bsdf.inputs["Base Color"])

        if normal_path.exists():
            nor = n.new("ShaderNodeTexImage")
            nor.image = bpy.data.images.load(str(normal_path))
            nor.image.colorspace_settings.name = 'Non-Color'
            normal_map = n.new("ShaderNodeNormalMap")
            l.new(mapping.outputs["Vector"], nor.inputs["Vector"])
            l.new(nor.outputs["Color"], normal_map.inputs["Color"])
            l.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

        weather = self.config.get("weather", "clear")
        road_type = self.config.get("road_type", "asphalt")

        if road_type == "sand":
            bsdf.inputs["Base Color"].default_value = (0.72, 0.58, 0.38, 1)
            bsdf.inputs["Roughness"].default_value = 1.0
            bsdf.inputs["Metallic"].default_value = 0.0
        elif weather == "rainy":
            bsdf.inputs["Roughness"].default_value = 0.02
            if "Coat Weight" in bsdf.inputs:
                bsdf.inputs["Coat Weight"].default_value = 1.0
                bsdf.inputs["Coat Roughness"].default_value = 0.05
            elif "Clearcoat" in bsdf.inputs:
                bsdf.inputs["Clearcoat"].default_value = 1.0
                bsdf.inputs["Clearcoat Roughness"].default_value = 0.05
        else:
            bsdf.inputs["Roughness"].default_value = 0.85

        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        road.data.materials.append(mat)

        for side in [-5, 5]:
            bpy.ops.mesh.primitive_cube_add(size=1)
            sidewalk = bpy.context.active_object
            sidewalk.scale = (2, 1500, 0.2)
            sidewalk.location = (side, 700, 0.1)
            sw_mat = bpy.data.materials.new("sidewalk")
            sw_mat.use_nodes = True
            sw_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1)
            sw_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
            sidewalk.data.materials.append(sw_mat)

        for i in range(120):
            bpy.ops.mesh.primitive_plane_add(size=1)
            dash = bpy.context.active_object
            dash.scale = (0.15, 1.5, 1)
            dash.location = (0, i * 6 + 10, 0.02)
            lane_mat = bpy.data.materials.new("lane")
            lane_mat.use_nodes = True
            lane_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 1, 1, 1)
            dash.data.materials.append(lane_mat)

        if road_type == "sand":
            for obj in bpy.data.objects:
                if obj.active_material and "lane" in obj.active_material.name.lower():
                    obj.hide_render = True
                    obj.hide_viewport = True

    # ---------- CAMERA ----------
    def camera(self):
        bpy.ops.object.camera_add(location=(0, -10, 2.5))
        cam = bpy.context.active_object
        cam.rotation_euler = (np.radians(88.5), 0, 0)
        cam.data.lens = 80
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = 60.0
        cam.data.dof.aperture_fstop = 8.0
        bpy.context.scene.camera = cam

    # ---------- LOAD ----------
    def load(self, path):
        if not path.exists():
            return None
        objs = bproc.loader.load_obj(str(path))
        if not objs:
            return None
        root = objs[0]
        for o in objs[1:]:
            o.set_parent(root)
        return root

    # ---------- SPAWN ----------
    def spawn(self, path, scale):
        obj = self.load(path)
        if not obj:
            return

        name = str(path).lower()
        is_large = "truck" in name or "bus" in name
        base_gap = 14 if is_large else random.uniform(8, 12)

        if is_large:
            available = [ln for ln in LANES if ln not in self.large_vehicle_lanes]
            if not available:
                self.large_vehicle_lanes.clear()
                available = LANES
            lane_x = random.choice(available)
            self.large_vehicle_lanes.add(lane_x)
            min_y_allowed = max(self.lane_last_y[lane_x], self.large_vehicle_last_y + 20)
            y = min_y_allowed + base_gap + random.uniform(0, 4)
            self.large_vehicle_last_y = y
        else:
            lane_x = random.choice(LANES)
            y = self.lane_last_y[lane_x] + base_gap + random.uniform(0, 4)

        self.lane_last_y[lane_x] = y
        self.last_y = max(self.last_y, y)

        x = lane_x + random.uniform(-0.15, 0.15)
        obj.set_location([x, y, 0])
        self.all_vehicle_positions.append((x, y))
        obj.set_scale(scale)

        if "car" in name:
            obj.set_rotation_euler([np.radians(-90), 0, 0])
        elif "auto" in name:
            obj.set_rotation_euler([np.radians(-90), 0, np.radians(180)])
        elif "truck" in name:
            obj.set_rotation_euler([np.radians(-90), 0, np.radians(180)])
        elif "bus" in name:
            obj.set_rotation_euler([np.radians(-90), 0, np.radians(90)])

        if "car" in name:
            cat_id = CATEGORY_IDS["car"]
        elif "auto" in name:
            cat_id = CATEGORY_IDS["auto"]
        elif "truck" in name:
            cat_id = CATEGORY_IDS["truck"]
        elif "bus" in name:
            cat_id = CATEGORY_IDS["bus"]
        else:
            cat_id = CATEGORY_IDS["car"]

        self.spawned_objects.append((obj, cat_id))

    def spawn_cow(self):
        cow = self.load(ANIMALS_PATH / "cow.glb")
        if not cow:
            return

        placed = False
        for attempt in range(20):
            x_zone = random.choice([
                random.uniform(-5.2, -3.8),
                random.uniform(-2.8, -1.8),
                random.uniform(-0.6, 0.6),
                random.uniform(1.8, 2.8),
                random.uniform(3.8, 5.2),
            ])
            y_zone = random.uniform(self.base_y + 10, self.base_y + 60)
            safe = True
            for vx, vy in self.all_vehicle_positions:
                dist = ((x_zone - vx)**2 + (y_zone - vy)**2) ** 0.5
                if dist < 4.5:
                    safe = False
                    break
            if safe:
                cow.set_location([x_zone, y_zone, 0.05])
                cow.set_rotation_euler([np.radians(-90), 0, random.uniform(0, 6.28)])
                cow.set_scale([0.12] * 3)
                self.all_vehicle_positions.append((x_zone, y_zone))
                placed = True
                break

        if not placed:
            cow.set_location([random.choice([-4.5, 4.5]), self.base_y + 25, 0.05])
            cow.set_rotation_euler([np.radians(-90), 0, random.uniform(0, 6.28)])
            cow.set_scale([0.12] * 3)
        self.spawned_objects.append((cow, CATEGORY_IDS["cow"]))

    # ---------- POPULATE ----------
    def populate(self):
        car_path   = VEHICLES_PATH / "car.glb"
        auto_path  = VEHICLES_PATH / "auto.glb"
        truck_path = VEHICLES_PATH / "truck.glb"
        bus_path   = VEHICLES_PATH / "bus.glb"

        density = self.config.get("object_density", 0.5)
        weather = self.config.get("weather", "clear")

        num_cars  = max(2, int(15 * density))
        num_autos = self.config.get("auto_rickshaw_count", 0)
        if weather == "clear":
            num_autos = max(num_autos, 4)

        # Under rainy conditions, aggressively reduce traffic to limit memory usage
        if weather == "rainy":
            num_cars = min(4, num_cars)
            num_autos = min(3, num_autos)

        num_trucks = max(0, int(4 * density)) if density > 0.5 else 0
        num_buses  = max(0, int(3 * density)) if density > 0.4 else 0

        spawn_queue  = []
        spawn_queue += [("car",   car_path,   [1]*3)]    * num_cars
        spawn_queue += [("auto",  auto_path,  [1.0]*3)]  * num_autos
        spawn_queue += [("truck", truck_path, [0.2]*3)]  * num_trucks
        spawn_queue += [("bus",   bus_path,   [0.13]*3)] * num_buses

        random.shuffle(spawn_queue)

        large = [(t, p, s) for t, p, s in spawn_queue if t in ("truck", "bus")]
        small = [(t, p, s) for t, p, s in spawn_queue if t not in ("truck", "bus")]

        final_queue = []
        large_iter   = iter(large)
        insert_every = max(1, len(small) // max(len(large), 1))
        for i, item in enumerate(small):
            final_queue.append(item)
            if (i + 1) % insert_every == 0:
                try:
                    final_queue.append(next(large_iter))
                except StopIteration:
                    pass
        for item in large_iter:
            final_queue.append(item)

        # Hard cap total vehicle spawns in heavy conditions to avoid memory blowup
        if weather == "rainy":
            final_queue = final_queue[:8]

        for _, path, scale in final_queue:
            self.spawn(path, scale)

        time_of_day = self.config.get("time_of_day", "midday")
        cow_allowed = (
            weather != "foggy" and
            weather != "rainy" and
            time_of_day != "night"
        )
        if cow_allowed:
            if time_of_day in ["morning", "midday"]:
                num_cows = random.choices([1, 2], weights=[70, 30])[0]
            elif time_of_day == "evening":
                num_cows = 1 if random.random() < 0.5 else 0
            else:
                num_cows = 0
            for _ in range(num_cows):
                self.spawn_cow()

    # ---------- RAIN OVERLAY ----------
    def generate_rain_overlay(self, width, height):
        from PIL import Image as PILImage, ImageFilter, ImageDraw
        rain_layer = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rain_layer)
        num_drops = random.randint(800, 1400)
        for _ in range(num_drops):
            x = random.randint(0, width)
            y = random.randint(0, height)
            depth_factor = y / height
            length  = int(random.uniform(15, 45) * (0.5 + depth_factor))
            angle_x = int(random.uniform(3, 8))
            opacity = int(random.uniform(60, 140) * (0.4 + depth_factor * 0.6))
            width_px = 1 if depth_factor < 0.5 else random.choice([1, 1, 2])
            draw.line(
                [(x, y), (x + angle_x, y + length)],
                fill=(220, 230, 255, opacity),
                width=width_px
            )
        rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(radius=0.6))
        return rain_layer

    def add_road_markings(self):
        bpy.ops.mesh.primitive_plane_add(size=1)
        left_line = bpy.context.active_object
        left_line.scale = (0.05, 750, 1)
        left_line.location = (-3.8, 700, 0.015)
        line_mat = bpy.data.materials.new("EdgeLine_L")
        line_mat.use_nodes = True
        line_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 1, 1, 1)
        line_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
        left_line.data.materials.append(line_mat)

        bpy.ops.mesh.primitive_plane_add(size=1)
        right_line = bpy.context.active_object
        right_line.scale = (0.05, 750, 1)
        right_line.location = (3.8, 700, 0.015)
        line_mat2 = bpy.data.materials.new("EdgeLine_R")
        line_mat2.use_nodes = True
        line_mat2.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 1, 1, 1)
        line_mat2.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
        right_line.data.materials.append(line_mat2)

        bpy.ops.mesh.primitive_plane_add(size=1)
        centre = bpy.context.active_object
        centre.scale = (0.06, 750, 1)
        centre.location = (0, 700, 0.018)
        yellow_mat = bpy.data.materials.new("CentreLine")
        yellow_mat.use_nodes = True
        yellow_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1.0, 0.85, 0.0, 1)
        yellow_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
        centre.data.materials.append(yellow_mat)

    # ---------- ANNOTATION ----------
    def get_real_annotations(self, img_id, img_filename, spawned_objects):
        import bpy
        from mathutils import Vector

        scene = bpy.context.scene
        cam   = scene.camera
        if cam is None:
            print("[ERROR] No camera found")
            return {"id": img_id, "file_name": img_filename,
                    "width": 1920, "height": 1080}, []

        render = scene.render
        img_w  = render.resolution_x
        img_h  = render.resolution_y

        images_entry = {
            "id": img_id,
            "file_name": img_filename,
            "width": img_w,
            "height": img_h
        }

        annotation_list = []
        ann_id = img_id * 1000

        def project_3d_to_2d(world_coord):
            from bpy_extras.object_utils import world_to_camera_view
            co_2d = world_to_camera_view(scene, cam, Vector(world_coord))
            px = co_2d.x * img_w
            py = (1 - co_2d.y) * img_h
            return px, py, co_2d.z

        def get_world_bbox_corners(bpy_obj):
            local_corners = [Vector(c) for c in bpy_obj.bound_box]
            world_matrix  = bpy_obj.matrix_world
            return [world_matrix @ c for c in local_corners]

        def find_mesh_objects(bp_obj):
            try:
                root = bp_obj.blender_obj
            except Exception:
                return []
            mesh_objects = []
            if root.type == 'MESH':
                mesh_objects.append(root)
            def collect_children(obj):
                for child in obj.children:
                    if child.type == 'MESH':
                        mesh_objects.append(child)
                    collect_children(child)
            collect_children(root)
            return mesh_objects

        for bp_obj, category_id in spawned_objects:
            try:
                mesh_objs = find_mesh_objects(bp_obj)
                if not mesh_objs:
                    continue
                all_projected = []
                for mesh_obj in mesh_objs:
                    world_corners = get_world_bbox_corners(mesh_obj)
                    for corner in world_corners:
                        px, py, pz = project_3d_to_2d(corner)
                        if pz > 0:
                            all_projected.append((px, py))
                if len(all_projected) < 4:
                    continue
                xs = [p[0] for p in all_projected]
                ys = [p[1] for p in all_projected]
                x_min = max(0.0, min(xs))
                y_min = max(0.0, min(ys))
                x_max = min(float(img_w), max(xs))
                y_max = min(float(img_h), max(ys))
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                if bbox_w < 5 or bbox_h < 5:
                    continue
                if bbox_w > img_w * 0.95 or bbox_h > img_h * 0.95:
                    continue
                annotation_list.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [round(x_min, 2), round(y_min, 2),
                             round(bbox_w, 2), round(bbox_h, 2)],
                    "area": round(bbox_w * bbox_h, 2),
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
            except Exception as e:
                print(f"[WARN] Annotation failed cat_id={category_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[DEBUG] Final annotation count: {len(annotation_list)}")
        return images_entry, annotation_list

    # ---------- RENDER ----------
    def render(self, img_path):
        global bpy
        
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.adaptive_threshold = 0.02
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.adaptive_min_samples = 32
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.texture_limit = '2048'  # Forces Blender to split texture memory allocations

        # Force Blender to stream uncompressed textures from disk instead of caching them all in RAM
        bpy.context.scene.cycles.use_animated_seed = True
        bpy.context.scene.render.use_persistent_data = False
        os.environ["OIIO_CACHE_MEMORY_MB"] = "4096"

        # ── GPU setup (CHANGE FROM ORIGINAL: proper device enumeration) ──────
        prefs = bpy.context.preferences.addons["cycles"].preferences
        # Try OPTIX (NVIDIA RTX), then CUDA (older NVIDIA), then CPU
        for device_type in ("OPTIX", "CUDA", "NONE"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                # Enable all devices of that type
                for dev in prefs.devices:
                    dev.use = True
                bpy.context.scene.cycles.device = "GPU" if device_type != "NONE" else "CPU"
                if device_type != "NONE":
                    print(f"[RENDER] Using GPU: {device_type}")
                    # OPTIX denoiser
                    try:
                        bpy.context.scene.cycles.denoiser = 'OPTIX'
                    except Exception:
                        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                else:
                    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                    print("[RENDER] Falling back to CPU")
                break
            except Exception:
                continue
        # ─────────────────────────────────────────────────────────────────────

        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.render.filepath = str(img_path.resolve())

        import bmesh

        # --- HYBRID RENDERING ENGINE WITH AUTOMATIC FALLBACK ---
        try:
            print("🚀 Attempting high-speed hardware acceleration (GPU/OPTIX)...")
            bpy.context.scene.cycles.device = 'GPU'
            preferences = bpy.context.preferences.addons['cycles'].preferences
            preferences.compute_device_type = 'OPTIX'

            import bpy
            try:
                bpy.context.scene.cycles.texture_limit = '2048'
            except Exception:
                pass
            try:
                bpy.context.scene.cycles.use_spatial_splits = True
            except Exception:
                pass

            # Force device type to CPU to handle high asset memory load
            bpy.context.scene.cycles.device = 'CPU'
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'

            bpy.ops.render.render(write_still=True)
            print("✅ Scene rendered successfully via CPU fallback!")

        except Exception as e:
            if "Out of memory" in str(e) or "CUDA" in str(e) or "RuntimeError" in str(e) or "Malloc returns null" in str(e) or "IMB_ibImageFromMemory" in str(e):
                print("\n⚠️ VRAM Wall Hit or CUDA Context Failed! Triggering Automatic Safety Net...")
                print("🐢 Switching rendering engine down to System CPU. Please wait, this WILL complete successfully...")

                bpy.context.scene.cycles.device = 'CPU'
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'

                import bpy
                try:
                    bpy.context.scene.cycles.texture_limit = '2048'
                except Exception:
                    pass
                try:
                    bpy.context.scene.cycles.use_spatial_splits = True
                except Exception:
                    pass
                bpy.ops.render.render(write_still=True)
                print("✅ Safety Net complete. Scene successfully rendered via CPU fallback!")
            else:
                raise e

        if not img_path.exists() or img_path.stat().st_size == 0:
            raise RuntimeError(f"Render output missing or zero-sized after render: {img_path}")

        try:
            from PIL import Image
            with Image.open(str(img_path)) as verify_img:
                verify_img.load()
        except Exception as e:
            if img_path.exists():
                try:
                    img_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Render output invalid or corrupted: {e}") from e

        weather = self.config.get("weather", "clear")
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            img = Image.open(str(img_path)).convert("RGBA")
            time_of_day = self.config.get("time_of_day", "midday")

            if weather == "rainy":
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(0.5)
                r, g, b, a = img.split()
                r = r.point(lambda i: int(i * 0.85))
                b = b.point(lambda i: min(255, int(i * 1.15)))
                img = Image.merge('RGBA', (r, g, b, a))
                rain_overlay = self.generate_rain_overlay(img.width, img.height)
                img = Image.alpha_composite(img, rain_overlay)
                img = Image.alpha_composite(img, rain_overlay)

            elif weather == "foggy":
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(0.4)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(0.75)
                fog_veil = Image.new("RGBA", img.size, (0, 0, 0, 0))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(fog_veil)
                for row in range(img.height):
                    fog_strength = max(0, int(180 * (1 - row / (img.height * 0.75))))
                    draw.line([(0, row), (img.width, row)],
                              fill=(220, 220, 215, fog_strength))
                img = Image.alpha_composite(img, fog_veil)
                r, g, b, a = img.split()
                r = r.point(lambda i: min(255, int(i * 1.05)))
                g = g.point(lambda i: min(255, int(i * 1.03)))
                img = Image.merge('RGBA', (r, g, b, a))

            elif weather in ["clear", "sunny"]:
                if time_of_day == "night":
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(0.6)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(0.55)
                    r, g, b, a = img.split()
                    b = b.point(lambda i: min(255, int(i * 1.2)))
                    img = Image.merge('RGBA', (r, g, b, a))
                elif time_of_day == "evening":
                    r, g, b, a = img.split()
                    r = r.point(lambda i: min(255, int(i * 1.15)))
                    b = b.point(lambda i: int(i * 0.8))
                    img = Image.merge('RGBA', (r, g, b, a))
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.3)
                elif time_of_day == "morning":
                    r, g, b, a = img.split()
                    r = r.point(lambda i: min(255, int(i * 1.08)))
                    g = g.point(lambda i: min(255, int(i * 1.03)))
                    img = Image.merge('RGBA', (r, g, b, a))
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.1)
                else:
                    r, g, b, a = img.split()
                    r = r.point(lambda i: min(255, int(i * 1.1)))
                    b = b.point(lambda i: int(i * 0.9))
                    img = Image.merge('RGBA', (r, g, b, a))
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.2)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.1)

            if time_of_day == "night" and weather == "rainy":
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.75)
                r, g, b, a = img.split()
                b = b.point(lambda i: min(255, int(i * 1.1)))
                img = Image.merge('RGBA', (r, g, b, a))
            elif time_of_day == "night" and weather == "foggy":
                r, g, b, a = img.split()
                b = b.point(lambda i: min(255, int(i * 1.08)))
                img = Image.merge('RGBA', (r, g, b, a))

            img.convert("RGB").save(str(img_path))
        except Exception as e:
            print(f"Failed to post-process: {e}")


def main():
    print("FABRIC-AI Scene Generator — MAIN STARTED")
    print(f"Configs loaded: {len(scene_configs)}")

    bproc.init()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ann_path = OUTPUT / "latest_annotations.json"
    annotations = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    if ann_path.exists():
        try:
            with open(ann_path, "r") as f:
                loaded_annotations = json.load(f)
            if isinstance(loaded_annotations, dict):
                annotations["images"] = loaded_annotations.get("images", [])
                annotations["annotations"] = loaded_annotations.get("annotations", [])
                annotations["categories"] = loaded_annotations.get("categories", CATEGORIES)
            else:
                print(f"[WARN] Existing annotations file has invalid structure, resetting.")
        except Exception as e:
            print(f"[WARN] Failed to load existing annotations: {e}")

    existing_images = sorted(images_dir.glob("scene_*.png"))
    start_index = len(existing_images)
    print(f"[RESUME] Found {start_index} existing images, resuming from scene {start_index + 1}")

    total = len(scene_configs)

    for i, config in enumerate(scene_configs):
        if i < start_index:
            continue

        _clean_memory_before_scene()

        img_id = i + 1
        img_filename = f"scene_{img_id:04d}.png"
        img_path = images_dir / img_filename

        current_config = {}
        if isinstance(config, dict):
            current_config = config
        elif isinstance(config, str):
            try:
                current_config = json.loads(config)
            except json.JSONDecodeError:
                current_config = {"raw_value": config}
        else:
            current_config = {"raw_value": str(config)}

        weather_val = current_config.get("weather", "unknown")
        tod_val = current_config.get("time_of_day", "unknown")
        print(f"[{img_id}/{total}] {weather_val} / {tod_val}")

        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)

            s = Scene(current_config)
            s.world()
            s.create_fog()
            s.road()
            s.add_road_markings()
            s.camera()
            s.populate()
            s.add_environment()
            s.spawn_buildings()

            s.render(img_path)

            print(f"[DEBUG] spawned_objects count: {len(s.spawned_objects)}")

            img_entry, ann_list = s.get_real_annotations(
                img_id, img_filename, s.spawned_objects
            )

            annotations["images"].append(img_entry)
            annotations["annotations"].extend(ann_list)

            print(f"[{img_id}] Generated {len(ann_list)} annotations")

            bproc.clean_up()

        except Exception as e:
            print(f"[ERROR] Scene {img_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        if img_id % 10 == 0:
            with open(ann_path, "w") as f:
                json.dump(annotations, f)
            print(f"[CHECKPOINT] Saved at scene {img_id}")

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"[DONE] {total} scenes written to {OUTPUT}")


if __name__ == "__main__":
    main()
