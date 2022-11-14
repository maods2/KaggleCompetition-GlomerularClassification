epc=5
python main.py --model_name=AlexNet --training_mode=default --datasets=dataset_image_100x100 --epochs=$epc
python main.py --model_name=LeNet --training_mode=default --datasets=dataset_image_100x100 --epochs=$epc
python main.py --model_name=ResNet50 --training_mode=default --datasets=dataset_image_100x100 --epochs=$epc
python main.py --model_name=VGG16 --training_mode=default --datasets=dataset_image_100x100 --epochs=$epc
python main.py --model_name=Xception --training_mode=default --datasets=dataset_image_100x100 --epochs=$epc