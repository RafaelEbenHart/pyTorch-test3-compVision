import pandas as pd
import matplotlib.pyplot as plt
from compVisionModel0 import model0Result,totalTrainTimeModel0
from compVisionModel1 import model1Result,TotalTrainTimeModel1
from cnnModel2 import model2Result,totalTrainTimeModel2


compareResult = pd.DataFrame([model0Result,
                                model1Result,
                                model2Result])
print(compareResult)

compareResult["training_time"] = [totalTrainTimeModel0,
                                  TotalTrainTimeModel1,
                                  totalTrainTimeModel2]
print(compareResult)


plt.figure(figsize=(15,2))
compareResult.set_index("ModelName")["ModelAcc"].plot(kind="barh")
plt.xlabel("accucary(%)")
plt.ylabel("model")
plt.show()