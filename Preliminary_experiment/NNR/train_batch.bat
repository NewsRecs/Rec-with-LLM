@REM python make_test_negative.py


python make_behavior_parsed.py

python write_new_config.py current_log_pop CEL LSTUR 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
python write_new_config.py rev_current_log_pop CEL LSTUR 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py

python write_new_config.py current_log_pop CEL NAML 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
python write_new_config.py rev_current_log_pop CEL NAML 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py

python write_new_config.py current_log_pop CEL NRMS 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
python write_new_config.py rev_current_log_pop CEL NRMS 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py



python make_behavior_parsed.py

python write_new_config.py current_log_pop CEL LSTUR 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
python write_new_config.py rev_current_log_pop CEL LSTUR 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py

python write_new_config.py current_log_pop CEL NAML 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
python write_new_config.py rev_current_log_pop CEL NAML 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py

python write_new_config.py current_log_pop CEL NRMS 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
python write_new_config.py rev_current_log_pop CEL NRMS 8 36 Adressa_5w(type1) 20_ltlive False random 105
python train.py
