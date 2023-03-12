CUDA_VISIBLE_DEVICES=2 python src/run.py \
	--file_dir data/mmkb-datasets/FB15K_DB15K \
	--rate 0.2 \
	--seed 2020 \
	--lr .0005 \
	--epochs 1000 \
	--hidden_units "300,300,300" \
	--check_point 50  \
	--bsize 7500 
