"""Blender headless conversion worker script.

Run via:
blender --background --python blender_convert.py -- <input_file> <output_glb>
"""

import os
import sys

try:
    import bpy
except ImportError:
    print("Error: This script must be executed inside Blender via blender --background --python ...")
    sys.exit(1)


def clear_scene():
    """Remove default cube, lights, cameras."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def convert(input_path: str, output_path: str):
    clear_scene()

    ext = os.path.splitext(input_path)[1].lower()
    print(f"[BlenderWorker] Importing {input_path} (format: {ext})")

    if ext == ".obj":
        # Blender 4+ / 5+ fast C++ OBJ importer
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=input_path)
        else:
            bpy.ops.import_scene.obj(filepath=input_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=input_path)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=input_path)
    elif ext == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=input_path)
        else:
            bpy.ops.import_mesh.stl(filepath=input_path)
    else:
        raise ValueError(f"Unsupported format for Blender conversion: {ext}")

    mesh_count = len([obj for obj in bpy.data.objects if obj.type == "MESH"])
    print(f"[BlenderWorker] Imported {mesh_count} meshes. Exporting to GLB -> {output_path}")

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_yup=True,
        export_apply=True,
    )
    print(f"[BlenderWorker] GLB export complete: {output_path}")


if __name__ == "__main__":
    # Command line args after '--'
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python blender_convert.py -- <input> <output.glb>")
        sys.exit(1)

    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        print("Error: Missing input or output argument")
        sys.exit(1)

    input_file = args[0]
    output_file = args[1]

    try:
        convert(input_file, output_file)
        sys.exit(0)
    except Exception as exc:
        print(f"[BlenderWorker] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
