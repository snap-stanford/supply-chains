python register_data.py --csv_file ./data/daily_transactions_all_years.csv --dataset_name tgbl-supplychains --metric total_amount --dir ./TGB/tgb/datasets/tgbl_supplychains --logscale

#train the model
cd TGB
CUDA_VISIBLE_DEVICES=0 python examples/linkproppred/tgbl-supplychains/tgn.py --data "tgbl-supplychains" --num_run 1 --seed 1

#back everything up
mkdir tgb/datasets/archive-allyears
mv tgb/datasets/tgbl_supplychains/* tgb/datasets/archive-allyears
mv examples/linkproppred/tgbl-supplychains/saved_models/* tgb/datasets/archive-allyears/
mv examples/linkproppred/tgbl-supplychains/saved_results/* tgb/datasets/archive-allyears/
cd ..
