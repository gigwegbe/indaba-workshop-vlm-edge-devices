from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="./annotations",  # directory containing instances_train2017.json etc.
    save_dir="converted_cardd",              # output directory
    use_segments=False,                      # set True if segmentation masks are needed
    use_keypoints=False
)
