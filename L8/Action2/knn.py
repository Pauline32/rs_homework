from surprise import KNNBasic,KNNWithMeans,KNNWithZScore, KNNBaseline
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
#trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo1 = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'False'})
#algo1.fit(trainset)

algo2 = KNNWithZScore(k=50,sim_options = {'user_based': False, 'verbose': 'False'})
#algo2.fit(trainset)

algo3 = KNNBasic(k=50,sim_options = {'user_based': False, 'verbose': 'False'})
#algo3.fit(trainset)

algo4 = KNNBaseline(k=50,sim_options = {'user_based': False, 'verbose': 'False'})
#algo4.fit(trainset)

cross_validate(algo1,data, measures=['RMSE', 'MAE'],cv=3,verbose=True)
cross_validate(algo2,data, measures=['RMSE', 'MAE'],cv=3,verbose=True)
cross_validate(algo3,data, measures=['RMSE', 'MAE'],cv=3,verbose=True)
cross_validate(algo4,data, measures=['RMSE', 'MAE'],cv=3,verbose=True)




##uid = str(196)
##iid = str(302)
##
##pred = algo.predict(uid, iid)
##print(pred)
