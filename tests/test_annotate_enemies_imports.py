def test_annotation_editor_exposes_wind_detector_used_by_editor_loop():
    import label_yolo_captures

    assert callable(label_yolo_captures.move_annotation_bundle)
