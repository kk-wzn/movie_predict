# movie_predict

## 尝试了一波 发现layer=3的时候表现比较好，迭代次数设置成了5000次，应该有更好的条件，之后可以再多看几组

## 数据来源猫眼 https://piaofang.maoyan.com/rankings/year ，当前取了历史前11的数据 推断哪吒2的数据，今天把数据更新了一波，历史数据选取的时候，删除了部分诡异数据，类似于上映半年后再次上映的部分

### 5000次训练和对应的loss的折线图
![image](https://github.com/user-attachments/assets/3f6de38b-bc12-4c2e-9289-873a4794c613)
### 5000次训练的模型预测的结果
![image](https://github.com/user-attachments/assets/b2cf7935-7280-4b73-9258-bc5f77b6b03a)

### 最后结果，蓝色为预估曲线，红色为实际曲线，由于当前为根据前7天结果预测之后的票房，所以蓝色部分前7天为0，哪吒2太超出预期了，结果不是很准，整体趋势看起来还行，看上去学会了周末和周中的区别，不过实际情况中 有2.14情人节的特殊情况，这一点会导致较大误差

### layer=3，500次训练的模型预测的结果
![image](https://github.com/user-attachments/assets/9319b850-be23-4aad-b18a-ca94e94f01c8)



