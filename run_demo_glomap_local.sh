python demo_glomap_local.py \
    --input_dir data/composite/images/subset \
    --output_dir output2 \
    --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --retrieval_model checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
    --glomap_bin /usr/local/bin/glomap \
    --device cuda \
    --pixel_tol 5 \
    --conf_thr 1.001