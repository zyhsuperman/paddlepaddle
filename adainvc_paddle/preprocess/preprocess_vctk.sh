. vctk.config

if [ $stage -le 0 ]; then
    echo "Running Stage 0: Creating Datasets..."
    python3 make_datasets_vctk.py $raw_data_dir/wav48 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $sample_rate $n_utt_attr
    echo "Stage 0 completed."
fi

if [ $stage -le 1 ]; then
    echo "Running Stage 1: Reducing Dataset..."
    python3 reduce_dataset.py $data_dir/train.pkl $data_dir/train_$segment_size.pkl $segment_size
    echo "Stage 1 completed."
fi

if [ $stage -le 2 ]; then
    echo "Running Stage 2: Sampling Training Samples..."
    python3 sample_single_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
    echo "Stage 2 completed."
fi

if [ $stage -le 3 ]; then
    echo "Running Stage 3: Sampling Testing Samples..."
    python3 sample_single_segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python3 sample_single_segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
    echo "Stage 3 completed."
fi
