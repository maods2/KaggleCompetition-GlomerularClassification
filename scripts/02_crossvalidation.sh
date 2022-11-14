folds=10
epc=200
DA=True
python main.py --model_name=AlexNet --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds 
python main.py --model_name=LeNet --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds 
python main.py --model_name=ResNet50 --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds 
python main.py --model_name=VGG16 --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds 
python main.py --model_name=Xception --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds 

# python main.py --model_name=AlexNet --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds --data_augumentation=$DA
# python main.py --model_name=LeNet --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds --data_augumentation=$DA
# python main.py --model_name=ResNet50 --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds --data_augumentation=$DA
# python main.py --model_name=VGG16 --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds --data_augumentation=$DA
# python main.py --model_name=Xception --training_mode=cross-validation --datasets=dataset_image_100x100 --epochs=$epc --cross_validation_folds=$folds --data_augumentation=$DA