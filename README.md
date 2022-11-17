# itmo_cv_lab3
# Теоретическая база  
В данное работе представлены нейросети двух видов: ViT (vit_base_patch16_224_in21k[1]) и CNN(tf_efficientnetv2_m_in21k[2], resnetv2_50x1_bitm_in21k[3])  
ViT - адаптация архитектуры трансформеров, применяемой в задачах NLP, для задач компьютерного зрения. 
<img src="https://theaisummer.com/static/60c2cbe8edb845502c8bbc3b9e88791d/ee604/vision-transformer.png" width=50% height=50%>   

efficientnetv2 - модификация efficientnet, используется neural architecture search и модификация операции MBConv.
<img src="https://miro.medium.com/max/1204/1*DOf_v9EJuOhldHugkB8dAA.png" width=35% height=35%>   

resnetv2 - модификация resnet, используется модификация Residual Block  
<img src="https://static.packt-cdn.com/products/9781788629416/graphics/B08956_02_10.jpg" width=25% height=25%>   


# Описание разработанной системы
Была написана программа на языке Python, которая тестирует вышеперечисленные нейросети, подсчитывается top 1 accuracy и top 5 accuracy.


# Результаты  

| model                      | acc_1 | acc_5 | Memory usage (torch+model), MB | time, s |
|----------------------------|-------|-------|--------------------------------|---------|
| vit_base_patch16_224_in21k | 0.6   | 0.88  | 1073                           | 0.36    |
| tf_efficientnetv2_m_in21k  | 0.6   | 0.84  | 959                            | 1.66    |
| resnetv2_50x1_bitm_in21k   | 0.6   | 0.9   | 917                            | 0.62    |

# Выводы по работе
В результате экспериментов, было выяснено, что resnetv2_50x1_bitm_in21 обладает наибольшей точностью, потребляет меньше всего памяти и находится на втором месте по скорости работы. 

# Использованные источники
[1] https://arxiv.org/abs/2010.11929  
[2] https://arxiv.org/abs/2104.00298  
[3] https://arxiv.org/abs/1912.11370  
