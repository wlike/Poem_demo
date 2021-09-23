#python generate_poem.py os_model_ch_poem/pytorch_model.bin os_model_ch_poem/config.json os_model_ch_poem/vocab.txt 1
#python generate_poem.py model_jl_libai/'epoch=5-step=299.ckpt' os_model_ch_poem/config.json os_model_ch_poem/vocab.txt 1
python generate_poem.py model_jl/'epoch=0-step=49999.ckpt' os_model_ch_poem/config.json os_model_ch_poem/vocab.txt 1
# add delimeter in training data
#python generate_poem.py model_jl_libai/'epoch=8-step=224.ckpt' os_model_ch_poem/config.json os_model_ch_poem/vocab.txt 1
#python generate_poem.py model_jl/'epoch=0-step=49999.ckpt' os_model_ch_poem/config.json os_model_ch_poem/vocab.txt 1
