# generating head's mask.
# conda env adnerf
# python DataProcess/Gen_HeadMask.py --img_dir "train_data/cnn_imgs"

# generating 68-facial-landmarks by face-alignment, which is from 
# https://github.com/1adrianb/face-alignment
# conda env headnerf & adnerf
# python DataProcess/Gen_Landmark.py --img_dir "train_data/cnn_imgs"

# generating the 3DMM parameters
# conda env headnerf
python Fitting3DMM/FittingNL3DMM.py --img_size 512 \
                                    --intermediate_size 256  \
                                    --batch_size 9 \
                                    --img_dir "train_data/cnn_imgs"