#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1

#export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,patchinit=1,patchprog=1,autopilot=1,weipreload=0,kvcacheopt=1,progshareopt=1
#export DT_OPT=varsub=1,lxopt=1,opfusion=1,arithfold=1,dataopt=1,autopilot=1,kvcacheopt=1,progshareopt=1
unset DT_OPT
export HOST_ENDIAN_FORMAT=big

#export SENCORELETS=2
#export SENCORES=32
#export DEE_DUMP_GRAPHS=granite8b
#export TVM_INSTALL_DIR=/opt/sentient/tvm
export TOKENIZERS_PARALLELISM=false
#export DATA_PREC=fp16
#export SENLIB_DEVEL_CONFIG_FILE=/root/.senlib.json
#export AIU_CONFIG_FILE_0=/root/.senlib.json
#export FLEX_RDMA_PCI_BUS_ADDR_0="0000:00:00.0"
#export AIU_WORLD_RANK_0="0000:00:00.0"
#export _MODEL=/models/granite-8b-code-instruct-unsafe
export DTLOG_LEVEL=error
#unset TORCH_SENDNN_LOG
export DT_DEEPRT_VERBOSE=-1
#export COMPILATION_MODE=offline_decoder
export DTCOMPILER_KEEP_EXPORT=true
#export FLEX_COMPUTE=SENTIENT
#export FLEX_DEVICE=VFIO
#export FLEX_OVERWRITE_NMB_FRAME=1
#export FLEX_MONITOR_SLEEP=0
#unset FMS_SKIP_TP_EMBEDDING


MODEL_PATH=/models/bert-base-uncased
DATASET_PATH=../twitter_complaints.json

python3 -u ./train_classification.py \
	--architecture=bert_classification \
	--variant=base \
	--num_classes=2 \
	--checkpoint_format=hf \
	--model_path=${MODEL_PATH}/pytorch_model.bin \
	--tokenizer=${MODEL_PATH} \
	--device_type=cpu \
	--dataset_style=sentiment \
	--dataset_path=${DATASET_PATH} \
	--compile \
	--head_only \
	--batch_size=2
