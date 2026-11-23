"""Blender Automated 3D Bridge Modeling & GLTF/GLB Exporter Script.

Run via Blender command line (headless background mode):
    blender --background --python backend/scripts/export_blender_bridge.py -- [output_dir]

Automates:
1. Clearing default scene objects.
2. Modeling low-poly 3-span bridge geometry matching ML channel node coordinates.
3. Outliner naming & custom property tagging (Sensor_Abutment_Left, Sensor_Pier_Left, etc.).
4. Baking transforms, setting origins to geometry.
5. Exporting optimized binary GLTF/GLB asset (`bridge_digital_twin.glb`).
"""

import sys
from pathlib import Path

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


SENSOR_NODES = [
    {
        "name": "Sensor_Abutment_Left",
        "type": "abutment",
        "loc": (-12.0, 0.0, 2.0),
        "dim": (2.2, 4.4, 4.0),
        "color": (0.44, 0.46, 0.48, 1.0)
    },
    {
        "name": "Sensor_Pier_Left",
        "type": "pier",
        "loc": (-4.0, 0.0, 2.0),
        "dim": (1.6, 4.0, 4.0),
        "color": (0.44, 0.46, 0.48, 1.0)
    },
    {
        "name": "Sensor_MidSpan_Girder",
        "type": "girder",
        "loc": (0.0, 0.0, 4.3),
        "dim": (26.0, 3.6, 0.6),
        "color": (0.29, 0.33, 0.41, 1.0)
    },
    {
        "name": "Sensor_Pier_Right",
        "type": "pier",
        "loc": (4.0, 0.0, 2.0),
        "dim": (1.6, 4.0, 4.0),
        "color": (0.44, 0.46, 0.48, 1.0)
    },
    {
        "name": "Sensor_Abutment_Right",
        "type": "abutment",
        "loc": (12.0, 0.0, 2.0),
        "dim": (2.2, 4.4, 4.0),
        "color": (0.44, 0.46, 0.48, 1.0)
    },
    {
        "name": "Sensor_Ambient_Temp",
        "type": "weather_station",
        "loc": (-12.0, 0.0, 5.2),
        "dim": (0.5, 0.5, 0.5),
        "color": (0.19, 0.59, 0.58, 1.0)
    }
]


def create_material(name, rgba_color, roughness=0.5, metallic=0.1):
    """Create a Principled BSDF material in Blender."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba_color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
    return mat


def build_and_export_bridge(output_glb_path: str):
    """Clear scene, construct bridge mesh tree, tag nodes, and export GLB."""
    if not IN_BLENDER:
        print("[BlenderBridgeExporter] Not running inside Blender environment.")
        return

    # 1. Clear default scene objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Collection for bridge
    bridge_coll = bpy.data.collections.new("SHM_Digital_Twin_Bridge")
    bpy.context.scene.collection.children.link(bridge_coll)

    # 2. Build tagged structural meshes
    created_objects = []

    for node in SENSOR_NODES:
        name = node["name"]
        loc = node["loc"]
        dim = node["dim"]
        color = node["color"]

        if node["type"] == "weather_station":
            # Sphere geometry for weather station
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=dim[0]/2.0, location=loc
            )
        else:
            # Box geometry for piers, abutments, girders
            bpy.ops.mesh.primitive_cube_add(location=loc)

        obj = bpy.context.active_object
        obj.name = name
        obj.data.name = f"{name}_Mesh"

        if node["type"] != "weather_station":
            # Cube is created at scale 1 with radius 1 (total size 2), so scale to dim/2
            obj.scale = (dim[0] / 2.0, dim[1] / 2.0, dim[2] / 2.0)
            bpy.ops.object.transform_apply(scale=True)

        # Set origin to geometry center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

        # Create and assign material
        mat = create_material(f"Mat_{name}", color)
        obj.data.materials.append(mat)

        # Attach custom metadata properties
        obj["sensor_id"] = name
        obj["sensor_type"] = node["type"]
        obj["shm_tagged"] = True

        created_objects.append(obj)

    # 3. Add ground slab (non-tagged decorative element)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, -0.2))
    ground = bpy.context.active_object
    ground.name = "Ground_Pad"
    ground.scale = (13.0, 3.0, 0.2)
    bpy.ops.object.transform_apply(scale=True)
    mat_ground = create_material("Mat_Ground", (0.2, 0.2, 0.2, 1.0), roughness=0.9)
    ground.data.materials.append(mat_ground)

    # 4. Select all objects and export GLTF/GLB
    bpy.ops.object.select_all(action='SELECT')

    out_file = Path(output_glb_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[BlenderBridgeExporter] Exporting GLB to: {out_file}")

    bpy.ops.export_scene.gltf(
        filepath=str(out_file),
        export_format='GLB',
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_custom_properties=True,
        export_materials='EXPORT',
    )

    print("[BlenderBridgeExporter] GLB export completed successfully.")


if __name__ == "__main__":
    if IN_BLENDER:
        # Parse command line args after '--'
        args = sys.argv
        output_path = "backend/artifacts/bridge_digital_twin.glb"
        if "--" in args:
            idx = args.index("--")
            if len(args) > idx + 1:
                output_path = args[idx + 1]
        build_and_export_bridge(output_path)
    else:
        print("[INFO] Script designed for running inside Blender via:")
        print("  blender --background --python backend/scripts/export_blender_bridge.py -- artifacts/bridge_digital_twin.glb")
