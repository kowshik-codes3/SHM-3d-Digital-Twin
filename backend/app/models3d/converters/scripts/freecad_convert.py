"""FreeCAD headless script to tessellate STEP / IGES geometry to GLB / mesh.

Run via:
FreeCADCmd freecad_convert.py <input_step_or_iges> <output_glb_or_obj> [tessellation_deflection]
"""

import os
import sys

def convert():
    argv = sys.argv
    if "--pass" in argv:
        args = argv[argv.index("--pass") + 1 :]
    else:
        args = argv[1:]

    if len(args) < 2:
        print("Usage: FreeCADCmd freecad_convert.py --pass <input> <output> [deflection]")
        sys.exit(1)

    input_path = os.path.abspath(args[0])
    output_path = os.path.abspath(args[1])
    deflection = float(args[2]) if len(args) > 2 else 0.02

    print(f"[FreeCADWorker] Converting {input_path} -> {output_path} (deflection: {deflection})")

    try:
        import FreeCAD
        import Part
        import MeshPart
        import Mesh

        ext = os.path.splitext(input_path)[1].lower()
        shape = Part.Shape()

        if ext in (".step", ".stp"):
            shape.read(input_path)
        elif ext in (".iges", ".igs"):
            import IGESControl
            shape.read(input_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        print(f"[FreeCADWorker] CAD shape read successfully. Tessellating with deflection {deflection}...")
        mesh = MeshPart.meshFromShape(shape, Deflection=deflection)
        
        print(f"[FreeCADWorker] Mesh generated with {mesh.CountFacets} triangles and {mesh.CountPoints} vertices")

        # Export intermediate OBJ or direct output
        mesh.write(output_path)
        print(f"[FreeCADWorker] Mesh successfully exported to {output_path}")

        sys.exit(0)
    except Exception as exc:
        print(f"[FreeCADWorker] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

convert()
