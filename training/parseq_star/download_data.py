import gdown, sys, hashlib, pathlib
Z = pathlib.Path('data/zips')
items = {
 'sha256sums_main.txt':'1_2fXopax0KpfBjH0VJVEzn4KYJHbWxMP', 'extract-datasets.sh':'1cXyHlQ8qszFOx6u0uwu3qApmZ3y_IrDv',
 'train.zip':'1DnWLcHgdD6xLE_l606pcUhKj07bAKC4S', 'val.zip':'1k37Xvk40F0U6DEGPhNurltOvkkaO6hAK', 'test.zip':'1qO1d3-3dIwQVZ-oBcTxCYEMuxbSPG7Pq',
 'sha256sums_textocr.txt':'14qkcFXNx9GqGxp7Ecowahn6JiakG4vfI', 'TextOCR_train.zip':'1jVjJFno4pnsU0Cp_kn4MIXQrChmELy92', 'TextOCR_val.zip':'1ubIRu01MXIek6OvInu-XjaIbw6277-vw',
 'OpenVINO_train_1.zip':'1q23QAIRTyG0t-bBm4aAwRwiqB6VUfphw', 'OpenVINO_train_2.zip':'1AtbaJljM68cbZqi5lcM92d9VkQUCbSqI', 'OpenVINO_train_5.zip':'1dejstYnJ8_sESuO_uvwi__jT1B8gPxf3',
 'OpenVINO_train_f.zip':'1C4akchTc7-yi1OS_sJ3KP693UKcnecke', 'OpenVINO_validation.zip':'17TRzSQhuK_juAxAv3KmX0y13pQP2cz6R',
}
for name, i in items.items():
    out = Z / name
    if out.exists() and out.stat().st_size > 0 and not name.endswith('.part'):
        print('have', name, flush=True); continue
    print('get', name, flush=True)
    gdown.download(id=i, output=str(out), quiet=True, use_cookies=False)
    print('done', name, out.stat().st_size, flush=True)
print('ALL DONE', flush=True)
