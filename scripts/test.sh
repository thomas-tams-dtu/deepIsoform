net_size=small
latent_features=(16 32 64 128 256 512 1024)
weight_decays=(1e-4 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
beta_values=(0.01 0.05 0.1 0.5 1 5 10 50)

if [ ${#latent_features[@]} -ne ${#weight_decays[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

for ((i=0; i<${#latent_features[@]}; i++)); do
    lf=${latent_features[i]}
    wd=${weight_decays[i]}

    for beta in "${beta_values[@]}"; do
        echo ${net_size}    ${lf}  ${wd}    ${beta}
        #/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/encoder_dense_train.py -ns ${net_size} -lf ${lf} -wd ${wd} -b ${b} -bs 500 -lr 1e-4 -p 6 -e 100
    done

done
