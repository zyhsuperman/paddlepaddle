# You can modify the bacth_size (e.g., 96, 128) to get similar results.

python3 main_trans.py --audio_dir=./feats/vggish/ --audio_enc=1 --audio_smoothing=1 --aug_type=ada \
--augment=1 --batch_size=64 --before_audio_smoothing=1 --before_vis_smoothing=0.9 --decay=0.5 --decay_epoch=7 \
--delta=0.85 --forward_dim=512 --init_epoch=2 --is_a_ori=0 --is_v_ori=0 --lr=7e-05 --mode=train --num_head=1 \
--num_layer=1 --occ_dim=128 --prob_drop=0.4 --prob_drop_occ=0.25 --st_dir=./feats/r2plus1d_18/ --tmp=1 --tsne=0 \
--video_dir=./feats/res152/ --vis_smoothing=1 --early_stop=10 --model_save_dir=./ckpt/
