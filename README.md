
usage:

cd src

step 1-4:
python run_pipeline.py --dataset ../dataset --output ../output --prep

step 5+7:
python run_pipeline.py --dataset ../dataset --output ../output --train_models 

step 6+8
python run_pipeline.py --dataset ../dataset --output ../output --train_models --scene ../dataset/garden --n_views "A,B,C"

