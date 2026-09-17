"""Checks for label errors that would silently change the training task."""
import unittest
from types import SimpleNamespace
import numpy as np
from geometry import project, camera_delta_cm
from generate import arc_sample, segment_subtasks

class GeometryTests(unittest.TestCase):
    def test_camera_convention_and_behind_camera(self):
        xy, valid = project(np.array([[0,0,-1],[.5,.5,-1],[0,0,1]]),np.zeros(3),np.eye(3),90)
        np.testing.assert_allclose(xy[:2],[[.5,.5],[.75,.25]])
        np.testing.assert_equal(valid,[True,True,False])

    def test_3d_axes_match_displayed_camera(self):
        np.testing.assert_allclose(camera_delta_cm([[1,-1,-1]],np.zeros(3),np.eye(3)),[[100,100,100]])

    def test_stationary_object_stays_five_points(self):
        p=arc_sample([[.2,.3],[.2,.3]])
        np.testing.assert_allclose(p,np.tile([.2,.3],(5,1)))

    def test_arc_length_not_time_sampling(self):
        p=arc_sample([[0,0],[.01,0],[.02,0],[1,0]])
        np.testing.assert_allclose(p[:,0],np.linspace(0,1,5))

    def test_two_doors_get_separate_full_windows(self):
        m=SimpleNamespace(body_jntadr=np.array([0,1]),body_jntnum=np.array([1,1]),jnt_qposadr=np.array([0,1]))
        states=np.zeros((30,3));states[:,1]=np.interp(np.arange(30),[0,5,10,29],[0,0,1,1]);states[:,2]=np.interp(np.arange(30),[0,20,25,29],[0,0,1,1])
        obj=np.zeros((30,2,3));obj[:,1,0]=1;grip=np.zeros((30,3));grip[15:,0]=1
        s=segment_subtasks(m,states,[0,1],obj,grip)
        self.assertEqual([(x['start'],x['end'],x['entity_index']) for x in s],[(0,15,0),(15,30,1)])
        self.assertTrue(all(x['boundary_needs_review'] for x in s))

    def test_tiny_contact_bump_does_not_merge_two_door_subtasks(self):
        m=SimpleNamespace(body_jntadr=np.array([0,1]),body_jntnum=np.array([1,1]),jnt_qposadr=np.array([0,1]))
        states=np.zeros((100,3));states[:,1]=np.interp(np.arange(100),[0,10,30,99],[0,0,1,1]);states[:,2]=np.interp(np.arange(100),[0,60,80,99],[0,0,1,1])
        states[90,1]+=.01
        obj=np.zeros((100,2,3));obj[:,1,0]=1;grip=np.zeros((100,3));grip[45:,0]=1
        sub=segment_subtasks(m,states,[0,1],obj,grip)
        self.assertEqual([(s['start'],s['end']) for s in sub],[(0,45),(45,100)])
        self.assertIn('central90pct',sub[0]['boundary_source'])

    def test_simultaneous_entities_are_not_given_invented_boundaries(self):
        m=SimpleNamespace(body_jntadr=np.array([0,1]),body_jntnum=np.array([1,1]),jnt_qposadr=np.array([0,1]))
        states=np.c_[np.zeros(30),np.linspace(0,1,30),np.linspace(0,1,30)]
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            segment_subtasks(m,states,[0,1],np.zeros((30,2,3)),np.zeros((30,3)))

if __name__=='__main__':unittest.main()
