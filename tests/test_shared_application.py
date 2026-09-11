import cv2
import numpy as np
import pytest
from shellshock.annotations.conversion import AnnotationBox,SceneAnnotation,annotations_to_world
from shellshock.perception.world import World

def test_manual_circle_is_authoritative_over_image_fit():
    image=np.zeros((200,200,3),np.uint8);cv2.circle(image,(95,95),35,(255,0,255),3)
    scene=SceneAnnotation(200,200,[AnnotationBox('obstacle_circle',45,45,100,100,source='manual')])
    world,_=annotations_to_world(scene,image)
    assert world.circles[0].radius==50
    assert world.circles[0].center==(95,95)

def test_common_solver_verifies_the_returned_angle_muzzle():
    from shellshock.application.solver import solve_integer_shot
    from shellshock.physics.launch import muzzle_position
    source=(100,700)
    result=solve_integer_shot(source,(850,700),World(image_width=1920),0,'right',1920,'normal_high')
    assert result['launch_point']==muzzle_position(source,result['direction'],result['angle_degrees'],1920)
    assert result['segments'][0]['start']==result['launch_point']

def test_overlay_uses_segments_and_breaks_portal_jump():
    from shellshock.rendering.trajectory import sample_solution_trajectory
    solution={'status':'reachable','segments':[{'start':(0,0),'velocity':(1,0),'acceleration':(0,0),'duration':1}, {'start':(100,0),'velocity':(1,0),'acceleration':(0,0),'duration':1}]}
    points=sample_solution_trajectory(solution,(0,0),1920)
    assert None in points
    assert points[0]==(0,0) and points[-1]==(101,0)

def test_exact_manual_circles_survive_save_reload(tmp_path):
    from shellshock.annotations.conversion import save_manual_scene,load_manual_scene
    scene=SceneAnnotation(200,200,[AnnotationBox('obstacle_circle',-40,30,100,100,source='manual')],self_center=(10,100),self_muzzle=(20,90))
    paths=tuple(tmp_path/x for x in ['label.txt','geo.json','meta.json'])
    save_manual_scene(scene,*paths)
    loaded=load_manual_scene(*paths,200,200)
    assert loaded.boxes[0].center==(10,80)
    assert loaded.boxes[0].width==100
    assert loaded.self_muzzle==(20,90)

def test_signed_wind_can_switch_from_left_to_right():
    from shellshock.application.scene import analyze_frame
    image=np.zeros((100,100,3),np.uint8);scene=SceneAnnotation(100,100)
    left=analyze_frame(image,scene=scene,wind_override=-5)
    right=analyze_frame(image,scene=scene,wind_override=5)
    assert left.wind_direction=='left' and right.wind_direction=='right'


def test_replay_diagnostics_show_current_stage_counts():
    from replay_shellshock import solver_log_lines
    lines = solver_log_lines({'diagnostics': {'layer_a_generated': 8, 'layer_a_rejected': 2,
        'layer_b_seed_count': 12, 'candidate_count': 50, 'verified_count': 3,
        'budget_exhausted': True, 'route_trace': [{'route': [('portal', 0)], 'layer_a': 'PASS', 'continuous_seeds': 4}]}})
    text = '\n'.join(lines)
    assert 'candidate_count=50' in text and 'budget_exhausted=True' in text
    assert 'portal' in text and 'PASS' in text

def test_manual_controls_preserve_solver_direction():
    from replay_shellshock import manual_controls
    scene = type('Scene', (), {'metadata': {'direction': 'right'}})()
    angle, power = manual_controls(scene, {'direction': 'left', 'angle_degrees': 0, 'power': 90})
    assert (angle, power) == (0, 90)
    assert scene.metadata['direction'] == 'left'
