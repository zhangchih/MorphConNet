# MorphConNet
Chi Zhang, Qihua Chen, Xuejin Chen. Self-Supervised Learning of Morphological Representation for 3D EM Segments with Cluster-Instance Correlations.
In MICCAI 2022
## Dataset: FAFB-CellSeg
```
cd fafb-cellseg
uzip *.zip 
```
The segment ids of corresponding classes are in glias/neurites/soma.txt. You can copy the ids and visualize them in [Neuroglancer](https://fafb-dot-neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B124024.3046875%2C58665.80859375%2C2292.3662109375%5D%2C%22crossSectionScale%22:4.779288037933938%2C%22projectionOrientation%22:%5B-0.5623587369918823%2C-0.446272075176239%2C-0.5405490398406982%2C0.438634991645813%5D%2C%22projectionScale%22:81567.93776427887%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig%22%2C%22name%22:%22fafb_v14%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe%22%2C%22name%22:%22fafb_v14_clahe%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://fafb-ffn1-20190805/segmentation%22%2C%22tab%22:%22segments%22%2C%22objectAlpha%22:0.23%2C%22segments%22:%5B%22710435991%22%5D%2C%22name%22:%22fafb-ffn1-20190805%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://gs://neuroglancer-20191211_fafbv14_buhmann2019_li20190805%22%2C%22tab%22:%22rendering%22%2C%22annotationColor%22:%22#cecd11%22%2C%22shader%22:%22#uicontrol%20vec3%20preColor%20color%28default=%5C%22blue%5C%22%29%5Cn#uicontrol%20vec3%20postColor%20color%28default=%5C%22red%5C%22%29%5Cn#uicontrol%20float%20scorethr%20slider%28min=0%2C%20max=1000%29%5Cn#uicontrol%20int%20showautapse%20slider%28min=0%2C%20max=1%29%5Cn%5Cnvoid%20main%28%29%20%7B%5Cn%20%20setColor%28defaultColor%28%29%29%3B%5Cn%20%20setEndpointMarkerColor%28%5Cn%20%20%20%20vec4%28preColor%2C%200.5%29%2C%5Cn%20%20%20%20vec4%28postColor%2C%200.5%29%29%3B%5Cn%20%20setEndpointMarkerSize%285.0%2C%205.0%29%3B%5Cn%20%20setLineWidth%282.0%29%3B%5Cn%20%20if%20%28int%28prop_autapse%28%29%29%20%3E%20showautapse%29%20discard%3B%5Cn%20%20if%20%28prop_score%28%29%3Cscorethr%29%20discard%3B%5Cn%7D%5Cn%5Cn%22%2C%22shaderControls%22:%7B%22scorethr%22:80%7D%2C%22linkedSegmentationLayer%22:%7B%22pre_segment%22:%22fafb-ffn1-20190805%22%2C%22post_segment%22:%22fafb-ffn1-20190805%22%7D%2C%22filterBySegmentation%22:%5B%22post_segment%22%2C%22pre_segment%22%5D%2C%22name%22:%22synapses_buhmann2019%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22n5://gs://fafb-v14-synaptic-clefts-heinrich-et-al-2018-n5/synapses_dt_reblocked%22%2C%22opacity%22:0.73%2C%22shader%22:%22void%20main%28%29%20%7BemitRGBA%28vec4%280.0%2C0.0%2C1.0%2CtoNormalized%28getDataValue%28%29%29%29%29%3B%7D%22%2C%22name%22:%22clefts_Heinrich_etal%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh#type=mesh%22%2C%22segments%22:%5B%221%22%2C%2210%22%2C%2211%22%2C%2212%22%2C%2213%22%2C%2214%22%2C%2215%22%2C%2216%22%2C%2217%22%2C%2218%22%2C%2219%22%2C%222%22%2C%2220%22%2C%2221%22%2C%2222%22%2C%2223%22%2C%2224%22%2C%2225%22%2C%2226%22%2C%2227%22%2C%2228%22%2C%2229%22%2C%223%22%2C%2230%22%2C%2231%22%2C%2232%22%2C%2233%22%2C%2234%22%2C%2235%22%2C%2236%22%2C%2237%22%2C%2238%22%2C%2239%22%2C%224%22%2C%2240%22%2C%2241%22%2C%2242%22%2C%2243%22%2C%2244%22%2C%2245%22%2C%2246%22%2C%2247%22%2C%2248%22%2C%2249%22%2C%225%22%2C%2250%22%2C%2251%22%2C%2252%22%2C%2253%22%2C%2254%22%2C%2255%22%2C%2256%22%2C%2257%22%2C%2258%22%2C%2259%22%2C%226%22%2C%2260%22%2C%2261%22%2C%2262%22%2C%2263%22%2C%2264%22%2C%2265%22%2C%2266%22%2C%2267%22%2C%2268%22%2C%2269%22%2C%227%22%2C%2270%22%2C%2271%22%2C%2272%22%2C%2273%22%2C%2274%22%2C%2275%22%2C%228%22%2C%229%22%5D%2C%22name%22:%22neuropil-regions-surface%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22mesh%22%2C%22source%22:%22vtk://https://storage.googleapis.com/neuroglancer-fafb-data/elmr-data/FAFB.surf.vtk.gz%22%2C%22shader%22:%22void%20main%28%29%20%7BemitRGBA%28vec4%281.0%2C%200.0%2C%200.0%2C%200.5%29%29%3B%7D%22%2C%22name%22:%22neuropil-full-surface%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%5B%7B%22url%22:%22precomputed://gs://fafb-ffn1-20190805/segmentation%22%2C%22subsources%22:%7B%22default%22:true%2C%22bounds%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22precomputed://gs://fafb-ffn1-20190805/segmentation/skeletons_32nm%22%5D%2C%22tab%22:%22segments%22%2C%22selectedAlpha%22:0%2C%22segments%22:%5B%22710435991%22%5D%2C%22segmentQuery%22:%22710435991%22%2C%22name%22:%22skeletons_32nm%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://fafb-ffn1/fafb-public-skeletons%22%2C%22tab%22:%22segments%22%2C%22segments%22:%5B%22710435991%22%5D%2C%22segmentQuery%22:%22710435991%22%2C%22name%22:%22public_skeletons%22%2C%22visible%22:false%7D%5D%2C%22showAxisLines%22:false%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22public_skeletons%22%2C%22visible%22:true%7D%2C%22layout%22:%22xy-3d%22%7D).

The point cloud data are in the folders, where each line records the coordinates of a point.

/*Note that we expanded the number of glia in the dataset to 331 (in the paper we have 110).  
Feature embedding space of MorphConNet on FAFB-CellSeg.
<div align="center"><img src="https://github.com/zhangchih/MorphConNet/blob/main/feature_distribution.png"  width="400px" />   

The glia segments we annotated in the fly brain:
<div align="center"><img src="https://github.com/zhangchih/MorphConNet/blob/main/glia_331.png"  width="800px" />   

<div align="left">  

## Test the pre-trained self-supervised model
```
uzip ./code/pretrained/model_sslcontrast.pth.zip
```
Finetune the model with the classification task of fafb-CellSeg: [finetune_eval.ipynb](https://github.com/zhangchih/MorphConNet/blob/main/code/eval/finetune_eval.ipynb)   
Linear classification and t-sne visualization: [linear_feature_eval.ipynb](https://github.com/zhangchih/MorphConNet/blob/main/code/eval/linear_feature_eval.ipynb)

## Pretrain the model with your own dataset
Format your dataset as:  
```
your_dataset_forlder  
├── neuron_dataset_shape_names.txt    # contains the class_name
├── neuron_dataset_test.txt   # contains the file names in folder class_name
├── class_name  
    ├── your_point_cloud_data1.txt  
    ├── your_point_cloud_data2.txt
    ├── ...
```
Modify the dataset path in ./main.py  
Config the training parameters in ./config/config.yaml
```
python main.py
```
