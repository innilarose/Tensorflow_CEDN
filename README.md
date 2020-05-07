# Tensorflow_CEDN
Tensorflow implementation of CEDN with custom dataset for object contour detection

To run train.py
python train.py --max_to_keep 50 --Epochs 10 --learning_rate .0000001 --train_crop_size 480 --train_text trrain_data.txt --log_dir log_directory --label_dir {path to label images} --image_dir {path to input image}

To run eval.py
python eval.py --checkpoint lo_directory/{checkpoint_file_naame} --save_preds predict --log_dir log_directory --eval_crop_size 480 --eval_text eval.txt --image_dir {path to test image}
