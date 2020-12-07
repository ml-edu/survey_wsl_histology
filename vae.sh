#python3 localization_mil.py -F results/vae_3 with dataset.glas dataset.fold=0 model.average model.arch='resnet18_vae'
#python3 localization_mil.py -F results/vae_3 with dataset.glas dataset.fold=1 model.average model.arch='resnet18_vae'
#python3 localization_mil.py -F results/vae_3 with dataset.glas dataset.fold=2 model.average model.arch='resnet18_vae'
#python3 localization_mil.py -F results/vae_3 with dataset.glas dataset.fold=3 model.average model.arch='resnet18_vae'
#python3 localization_mil.py -F results/vae_3 with dataset.glas dataset.fold=4 model.average model.arch='resnet18_vae'

python3 localization_mil.py -F results/vae_cams with dataset.glas dataset.fold=0 model.average model.arch='resnet18_vae' balance=0.0001
python3 localization_mil.py -F results/vae_cams with dataset.glas dataset.fold=0 model.average model.arch='resnet18_vae' balance=0