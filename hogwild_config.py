NUMBER_OF_DATA_SPLITED=30
NUMBER_OF_CLIENT=1

NUMBER_OF_MODEL_SPLITED=20
TAU=[1,3,5,10,15,20,50]
TAU_ARRAY=[]
NUMBER_OF_EPOCH=3
EPOCH_CHECKPOINT=10
RANDOM_SLEEP_TIME=3 # 1->20 * 0.1
history={'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]}

#plotting_info={"n_thread_type":[1,3,5,10,15],"loss_vals":[]}
#plotting_info={"n_thread_type":[1],"loss_vals":[],"info":[],"duration":[]}
#plotting_info={"n_thread_type":[15],"loss_vals":[],"info":[],"duration":[],"accuracy":[]}
plotting_info={"n_thread_type":[1],"loss_vals":[],"info":[],"duration":[],"accuracy":[]}
#plotting_info={"n_thread_type":[3,5,10,15],"loss_vals":[],"info":[],"duration":[],"accuracy":[]}
#plotting_info={"n_thread_type":[1,3,5,10,15,20 ],"loss_vals":[]}

# for i in range(0,len(plotting_info["n_thread_type"])):
#    plotting_info["loss_vals"].append([])
    # for j in range(NUMBER_OF_EPOCH):
    #     plotting_info["loss_vals"][i].append([])
    
        

# loss_val_in_epoch=[]
# for i in range(0,NUMBER_OF_EPOCH):
#     loss_val_in_epoch.append([])
loss_thread=[]
accuracy_thread=[]
for _ in range(NUMBER_OF_CLIENT):
    loss_thread.append([])
    accuracy_thread.append([])