"""Procedural parametric 3D model generator for bridges, buildings, and towers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh


class ProceduralModelGenerator:
    """Generates standard parametric GLB structures with semantically named parts."""

    @staticmethod
    def generate_girder_bridge(
        span_meters: float = 50.0,
        width_meters: float = 10.0,
        pier_count: int = 2,
        pier_height: float = 8.0,
        deck_thickness: float = 1.0,
    ) -> trimesh.Scene:
        """Create parametric girder bridge with Deck, Piers, and Abutments."""
        scene = trimesh.Scene()

        # 1. Deck
        deck_box = trimesh.creation.box(
            extents=[span_meters, deck_thickness, width_meters]
        )
        deck_y = pier_height + deck_thickness / 2.0
        deck_box.apply_translation([0.0, deck_y, 0.0])
        scene.add_geometry(deck_box, geom_name="Deck", node_name="Deck")

        # 2. Abutments
        abutment_width = 3.0
        abutment_box_left = trimesh.creation.box(
            extents=[abutment_width, pier_height, width_meters + 1.0]
        )
        abutment_box_left.apply_translation(
            [-span_meters / 2.0 + abutment_width / 2.0, pier_height / 2.0, 0.0]
        )
        scene.add_geometry(abutment_box_left, geom_name="Abutment_Left", node_name="Abutment_Left")

        abutment_box_right = trimesh.creation.box(
            extents=[abutment_width, pier_height, width_meters + 1.0]
        )
        abutment_box_right.apply_translation(
            [span_meters / 2.0 - abutment_width / 2.0, pier_height / 2.0, 0.0]
        )
        scene.add_geometry(abutment_box_right, geom_name="Abutment_Right", node_name="Abutment_Right")

        # 3. Intermediate Piers
        if pier_count > 0:
            pier_spacing = span_meters / (pier_count + 1)
            for i in range(1, pier_count + 1):
                px = -span_meters / 2.0 + i * pier_spacing
                pier_col = trimesh.creation.cylinder(
                    radius=1.2,
                    height=pier_height,
                    sections=24,
                )
                pier_col.apply_translation([px, pier_height / 2.0, 0.0])
                scene.add_geometry(pier_col, geom_name=f"Pier_{i}", node_name=f"Pier_{i}")

        # 4. Parapet barriers
        barrier_height = 0.9
        barrier_thick = 0.3
        b_left = trimesh.creation.box(extents=[span_meters, barrier_height, barrier_thick])
        b_left.apply_translation(
            [0.0, pier_height + deck_thickness + barrier_height / 2.0, -width_meters / 2.0 + barrier_thick / 2.0]
        )
        scene.add_geometry(b_left, geom_name="Barrier_North", node_name="Barrier_North")

        b_right = trimesh.creation.box(extents=[span_meters, barrier_height, barrier_thick])
        b_right.apply_translation(
            [0.0, pier_height + deck_thickness + barrier_height / 2.0, width_meters / 2.0 - barrier_thick / 2.0]
        )
        scene.add_geometry(b_right, geom_name="Barrier_South", node_name="Barrier_South")

        return scene

    @staticmethod
    def generate_building(
        floors: int = 4,
        bays_x: int = 3,
        bays_z: int = 2,
        floor_height: float = 3.5,
        bay_width: float = 6.0,
    ) -> trimesh.Scene:
        """Create parametric multi-story building with named floors, columns, and roof."""
        scene = trimesh.Scene()

        width_x = bays_x * bay_width
        depth_z = bays_z * bay_width
        slab_thick = 0.4

        # Foundation
        found = trimesh.creation.box(extents=[width_x + 2.0, 0.6, depth_z + 2.0])
        found.apply_translation([0.0, 0.3, 0.0])
        scene.add_geometry(found, geom_name="Foundation", node_name="Foundation")

        # Floors
        for f in range(1, floors + 1):
            floor_y = f * floor_height
            slab = trimesh.creation.box(extents=[width_x, slab_thick, depth_z])
            slab.apply_translation([0.0, floor_y - slab_thick / 2.0, 0.0])
            scene.add_geometry(slab, geom_name=f"Floor_{f}", node_name=f"Floor_{f}")

            # Columns supporting this level
            col_h = floor_height - slab_thick
            col_y = (f - 1) * floor_height + 0.6 + col_h / 2.0
            col_nodes = []
            for ix in range(bays_x + 1):
                cx = -width_x / 2.0 + ix * bay_width
                for iz in range(bays_z + 1):
                    cz = -depth_z / 2.0 + iz * bay_width
                    col = trimesh.creation.box(extents=[0.5, col_h, 0.5])
                    col.apply_translation([cx, col_y, cz])
                    col_nodes.append(col)

            merged_cols = trimesh.util.concatenate(col_nodes)
            scene.add_geometry(merged_cols, geom_name=f"Columns_Level_{f}", node_name=f"Columns_Level_{f}")

        # Roof
        roof_y = floors * floor_height + 0.2
        roof = trimesh.creation.box(extents=[width_x + 0.8, 0.4, depth_z + 0.8])
        roof.apply_translation([0.0, roof_y, 0.0])
        scene.add_geometry(roof, geom_name="Roof", node_name="Roof")

        return scene

    @staticmethod
    def generate_tower(
        height_meters: float = 36.0,
        base_width: float = 8.0,
        top_width: float = 3.0,
        sections: int = 4,
    ) -> trimesh.Scene:
        """Create parametric communications/observation tower."""
        scene = trimesh.Scene()

        # Foundation pedestal
        ped = trimesh.creation.box(extents=[base_width + 2.0, 1.0, base_width + 2.0])
        ped.apply_translation([0.0, 0.5, 0.0])
        scene.add_geometry(ped, geom_name="Foundation", node_name="Foundation")

        sec_height = height_meters / sections
        for s in range(sections):
            t0 = s / sections
            t1 = (s + 1) / sections
            w0 = base_width + (top_width - base_width) * t0
            w1 = base_width + (top_width - base_width) * t1
            y_base = 1.0 + s * sec_height

            # Create box approximation for section
            mid_w = (w0 + w1) / 2.0
            sec_box = trimesh.creation.box(extents=[mid_w, sec_height, mid_w])
            sec_box.apply_translation([0.0, y_base + sec_height / 2.0, 0.0])
            scene.add_geometry(sec_box, geom_name=f"Section_{s + 1}", node_name=f"Section_{s + 1}")

        # Platform
        plat_y = 1.0 + height_meters
        plat = trimesh.creation.cylinder(radius=top_width * 1.2, height=0.6, sections=32)
        plat.apply_translation([0.0, plat_y, 0.0])
        scene.add_geometry(plat, geom_name="Observation_Platform", node_name="Observation_Platform")

        return scene

    @classmethod
    def generate_for_structure(
        cls,
        structure_type: str,
        params: Optional[Dict[str, Any]] = None,
        output_glb_path: Optional[Path] = None,
    ) -> trimesh.Scene:
        """Generate parametric model based on structure type and export to GLB if path given."""
        params = params or {}
        stype = structure_type.lower()

        if "bridge" in stype:
            scene = cls.generate_girder_bridge(
                span_meters=params.get("span_meters", 50.0),
                width_meters=params.get("width_meters", 10.0),
                pier_count=params.get("pier_count", 2),
                pier_height=params.get("pier_height", 8.0),
            )
        elif "building" in stype:
            scene = cls.generate_building(
                floors=params.get("floors", 4),
                bays_x=params.get("bays_x", 3),
                bays_z=params.get("bays_z", 2),
            )
        elif "tower" in stype:
            scene = cls.generate_tower(
                height_meters=params.get("height_meters", 36.0),
            )
        else:
            # Default to girder bridge
            scene = cls.generate_girder_bridge()

        if output_glb_path:
            output_glb_path.parent.mkdir(parents=True, exist_ok=True)
            glb_data = scene.export(file_type="glb")
            output_glb_path.write_bytes(glb_data)

        return scene
