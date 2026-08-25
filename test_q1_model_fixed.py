# -*- coding: utf-8 -*-

import unittest

import numpy as np

from q1_model import sun_direction
from q1_model_fixed import (
    EvaluationConfig,
    NeighborIndex,
    RECEIVER_CENTER_Z,
    _ray_rectangle_hits,
    mirror_poses,
    midpoint_grid,
    ray_hits_receiver_cylinder,
    solar_disc_directions,
)


class FixedEvaluatorTests(unittest.TestCase):
    def test_receiver_center_uses_tower_height_definition(self):
        self.assertEqual(RECEIVER_CENTER_Z, 80.0)

    def test_midpoint_grid_is_inside_mirror_and_area_symmetric(self):
        aa, bb = midpoint_grid(6.0, 6.0, 7)
        self.assertEqual(len(aa), 49)
        self.assertLess(np.max(np.abs(aa)), 3.0)
        self.assertLess(np.max(np.abs(bb)), 3.0)
        self.assertAlmostEqual(float(np.mean(aa)), 0.0, places=12)
        self.assertAlmostEqual(float(np.mean(bb)), 0.0, places=12)

    def test_central_reflection_points_to_receiver(self):
        config = EvaluationConfig(sun_rays=1)
        sun, _ = sun_direction(80, 12.0)
        _, normals, _, _, reflected, _ = mirror_poses(
            np.array([150.0]), np.array([20.0]), sun, config
        )
        incoming = -sun
        outgoing = incoming - 2.0 * np.dot(incoming, normals[0]) * normals[0]
        outgoing /= np.linalg.norm(outgoing)
        self.assertTrue(np.allclose(outgoing, reflected[0], atol=1e-12))

    def test_receiver_cylinder_side_intersection(self):
        config = EvaluationConfig()
        origins = np.array([[[100.0, 0.0, 80.0]]])
        directions = np.array([[[-1.0, 0.0, 0.0]]])
        hits = ray_hits_receiver_cylinder(origins, directions, config)
        self.assertTrue(bool(hits[0, 0, 0]))

    def test_receiver_cylinder_does_not_count_inner_face_on_far_side(self):
        config = EvaluationConfig()
        origins = np.array([[[100.0, 0.0, 65.85]]])
        directions = np.array([[[-1.0, 0.0, 0.1]]])
        hits = ray_hits_receiver_cylinder(origins, directions, config)
        self.assertFalse(bool(hits[0, 0, 0]))

    def test_shadow_ray_must_point_toward_sun(self):
        points = np.array([[0.0, 0.0, 0.0]])
        candidate_centers = np.array([[10.0, 0.0, 0.0]])
        candidate_normals = np.array([[-1.0, 0.0, 0.0]])
        candidate_u = np.array([[0.0, 1.0, 0.0]])
        candidate_v = np.array([[0.0, 0.0, 1.0]])
        toward_sun = _ray_rectangle_hits(
            points,
            np.array([1.0, 0.0, 0.0]),
            candidate_centers,
            candidate_normals,
            candidate_u,
            candidate_v,
            1.0,
            1.0,
        )
        away_from_sun = _ray_rectangle_hits(
            points,
            np.array([-1.0, 0.0, 0.0]),
            candidate_centers,
            candidate_normals,
            candidate_u,
            candidate_v,
            1.0,
            1.0,
        )
        self.assertTrue(bool(toward_sun[0]))
        self.assertFalse(bool(away_from_sun[0]))

    def test_solar_disc_sampling_is_deterministic_and_bounded(self):
        sun = np.array([0.0, 0.6, 0.8])
        first = solar_disc_directions(sun, 32, 4.65e-3)
        second = solar_disc_directions(sun, 32, 4.65e-3)
        self.assertTrue(np.array_equal(first, second))
        angles = np.arccos(np.clip(first @ sun, -1.0, 1.0))
        self.assertLessEqual(float(np.max(angles)), 4.65e-3 + 1e-12)

    def test_neighbor_radius_is_monotone(self):
        xs = np.array([0.0, 20.0, 40.0, 80.0])
        ys = np.zeros_like(xs)
        small = NeighborIndex(xs, ys, radius=25.0)
        large = NeighborIndex(xs, ys, radius=45.0)
        self.assertTrue(np.all(large.counts >= small.counts))


if __name__ == "__main__":
    unittest.main()
