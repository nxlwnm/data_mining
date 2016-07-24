//
//  main.cpp
//  data_mining_4
//
//  Created by 张铭 on 16/5/5.
//  Copyright © 2016年 张铭. All rights reserved.
//
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <time.h>
#include <cmath>
#include <cmath> // for RAND, and rand

#define TIMES 1000
#define LAYER 4
#define NODE_EACH_LAYER {384, 180, 30, 1}
#define RATE 1
#define random (0.0001*sampleNormal())
/* 
200:1.85
500:0.8
20 60 8 1 10 ：7.019
20 50 6 1 10: 7.58
20 55 7 1 10:5.51
20 58 7 1 10：5.49
20 59 7 1 10: 8.82
20 58 8 1 10:5.69
20 58 6 1 10:7.87
20 58 7 1 5:5.49
 */

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

//神经网络节点
struct node {
    double output = 0;
    double err = 0;
    double bias = 0;
};
//神经网络边，第一维是第n层的边,第二三维分别是边起点，终点
vector<vector<vector<double> > > weigh;
//训练集，第一维是训练集个数，第二维是每个训练用例的384个参数
vector<vector<double> > training_set;
//训练集的结果
vector<double> ref;
vector<double> ref2;
//神经网络
vector<vector<node> > net;

double max, min;

void initialization();
void input(string);
void normalization();
double sampleNormal();

int main(int argc, const char * argv[]) {
    //从文件中输入训练集, 参数初始化(标准正态分布), 正则化
    input("/Users/zhangming/Documents/code/C/data_mining_4/data_mining_4/train.csv");
    initialization();
    normalization();
    
    
    for (int times = 0; times < TIMES; times++) {
        if(times%50 == 0) cout << "第" << times << "次训练\n";
        for (int i = 0; i < training_set.size(); i++) {
            int para[LAYER] = NODE_EACH_LAYER;
            //store the data into net
            for (int j = 0; j < para[0]; j++) net[0][j].output = training_set[i][j];
            
            //if (i == 0) for (int j = 0; j < para[0]; j++) cout << net[0][j].output << " ";
            
            //前向传播，求出output
            //计算每一层网络
            for (int j = 1; j < LAYER; j++) {
                //计算每一层网络中的每个节点
                for (int k = 0; k < para[j]; k++) {
                    double temp = 0;
                    //对每个节点，计算该节点与 上一层每个节点output和权值的积的和
                    for (int l = 0; l < para[j-1]; l++) temp += (net[j-1][l].output * weigh[j-1][l][k]);
                    temp += net[j][k].bias;
                    
                    //激活函数，可选用sigmoid 或者relu
                    net[j][k].output = 1/(1+exp(-temp));
                    //if (i == 0) cout << net[j][k].output << " ";
                }
            }
            //if (i ==0) cout << net[LAYER-1][0].output << endl;
            
            //后向传播
            //计算输出层的err
            for (int j = 0; j < para[LAYER-1]; j++) {
                //激活函数的导，使用relu后要改
                net[LAYER-1][j].err = net[LAYER-1][j].output * (1-net[LAYER-1][j].output) * (ref[i]-net[LAYER-1][j].output);
            }
            //if (i == 0) cout << ref[i] << " " << net[LAYER-1][0].err << endl;
            
            //计算中间每层的err
            for (int j = LAYER-2; j > 0; j--) {
                //计算每层的每个节点的err
                for (int k = 0; k < para[j]; k++) {
                    //对每个节点，计算该节点gz的导和（后一层的err和权值的积的和）的积
                    double temp = 0;
                    for (int l = 0; l < para[j+1]; l++) temp += (net[j+1][l].err * weigh[j][k][l]);
                    //激活函数的导，使用relu后要改
                    temp *= (net[j][k].output * (1-net[j][k].output));
                    net[j][k].err = temp;
                    //if (i == 0) cout << temp << endl;
                }
            }
            
            //调整权重
            //对LAYER－1层weigh
            for (int j = LAYER-2; j >= 0; j--) {
                //每一层的k*l矩阵
                for (int k = 0; k < para[j]; k++) {
                    for (int l = 0; l < para[j+1]; l++) {
                        weigh[j][k][l] += (RATE * net[j+1][l].err * net[j][k].output);
                        //if (i == 0 && j == 0 && k == 3) cout << weigh[j][k][l] << endl;
                    }
                }
            }
            //调整bias
            //对除第一层的每一层
            for (int j = 1; j < LAYER; j++) {
                //对每个节点
                for (int k = 0; k < para[j]; k++) {
                    net[j][k].bias += RATE * net[j][k].err;
                }
            }
            //if (i == 0) cout << net[LAYER-1][0].bias << endl;
        }
        
    }
    cout << "training completed\n";
    
    //test 方差
    double squre = 0;
    for (int i = 0; i < training_set.size(); i++) {
        int para[LAYER] = NODE_EACH_LAYER;
        for (int j = 0; j < para[0]; j++) net[0][j].output = training_set[i][j];
        //前向传播，求出output
        //计算每一层网络
        for (int j = 1; j < LAYER; j++) {
            //计算每一层网络中的每个节点
            for (int k = 0; k < para[j]; k++) {
                double temp = 0;
                //对每个节点，计算该节点与 上一层每个节点output和权值的积的和
                for (int l = 0; l < para[j-1]; l++) temp += (net[j-1][l].output * weigh[j-1][l][k]);
                temp += net[j][k].bias;
                //激活函数
                net[j][k].output = 1/(1+exp(-temp));
            }
        }
        double ans = net[LAYER-1][0].output * (max - min) +min;
        squre += pow(ref2[i] - ans ,2);
    }
    cout << "方差：" << squre/training_set.size() << endl;
    
    ifstream testin("/Users/zhangming/Documents/code/C/data_mining_4/data_mining_4/test.csv");
    if (!testin.good()) {
        cout << "file open failed\n";
        return 0;
    }
    ofstream testout("/Users/zhangming/Documents/code/C/data_mining_4/data_mining_4/testout.csv");
    testout << "Id,reference\n";
    int count = 0;
    string str, num;
    vector<double> templ;
    getline(testin, str);
    while (getline(testin, str)) {
        templ.clear(), num.clear();
        for (int i = int(str.find(",")+1); i < str.length(); i++) {
            if (str[i] == ',') {
                double num1 = atof(num.c_str());
                templ.push_back(num1);
                num.clear();
            } else num += str[i];
        }
        templ.push_back(double(atof(num.c_str())));
        
        int para[LAYER] = NODE_EACH_LAYER;
        for (int j = 0; j < para[0]; j++) net[0][j].output = templ[j];
        
        //前向传播，求出output
        //计算每一层网络
        for (int j = 1; j < LAYER; j++) {
            //计算每一层网络中的每个节点
            for (int k = 0; k < para[j]; k++) {
                double temp = 0;
                //对每个节点，计算该节点与 上一层每个节点output和权值的积的和
                for (int l = 0; l < para[j-1]; l++) temp += (net[j-1][l].output * weigh[j-1][l][k]);
                temp += net[j][k].bias;
                //激活函数
                net[j][k].output = 1/(1+exp(-temp));
            }
        }
        double ans = net[LAYER-1][0].output * (max - min) +min;
        testout << count << "," << ans << endl;
        
        count++;
    }
    
    return 0;
}

//生成参数，写入文件
void initialization() {
    /*
       384*100 100*20 20*1 parameter will store in para.ini and be read in a 3d-vector weigh
       384 100 20 1 node will store in a 2d-vector net, it need an attention that the first
    384 nodes' output will be the data in training_set, err will not be used, bias need compution
     */
    int para[LAYER] = NODE_EACH_LAYER;
    srand((unsigned int)time(0));
    ofstream para_out("/Users/zhangming/Documents/code/C/data_mining_4/data_mining_4/para.ini", 'w');
    for (int i = 0; i < LAYER-1; i++) {
        for (int j = 0; j < para[i]; j++) {
            for (int k = 0; k < para[i+1]; k++) {
                double random_num = random;
                para_out << random_num;
                if (k != para[i+1]-1) {
                    para_out << " ";
                } else para_out << "\n";
            }
        }
    }
    para_out.close();
    ifstream para_in("/Users/zhangming/Documents/code/C/data_mining_4/data_mining_4/para.ini", 'r');
    double num;
    for (int i = 0; i < LAYER-1; i++) {
        vector<vector<double> > temp_out;
        for (int j = 0; j < para[i]; j++) {
            vector<double> temp_in;
            for (int k = 0; k < para[i+1]; k++) {
                para_in >> num;
                temp_in.push_back(num);
            }
            temp_out.push_back(temp_in);
        }
        weigh.push_back(temp_out);
    }
    //cout << weigh.back().back().back();
    para_in.close();
    
    for (int i = 0; i < LAYER; i++) {
        vector<node> temp_node;
        for (int j = 0; j < para[i]; j++) {
            struct node new_node;
            temp_node.push_back(new_node);
        }
        net.push_back(temp_node);
    }
    
}

//正则化，把ref归一化到0-1，如果采用ReLu不用正则化
void normalization() {
    max = ref[0], min = ref[0];
    for (int i = 1; i < ref.size(); i++) {
        if (max < ref[i]) max = ref[i];
        if (min > ref[i]) min = ref[i];
    }
    for (int i = 0; i < ref.size(); i++) ref[i] = (ref[i] - min)/(max - min);
    
    //cout << max << " " << min << endl;
}

//从csv文件中输入到内存中
void input(string file_path) {
    /*
     input data in train set into 2d-vector traning_set
     the y, or the reference stored in vector ref and ref2
     ref will be used to normalization, ref2 will store the origin data
     */
    ifstream fin(file_path);
    if (!fin.good()) {
        cout << "file open failed\n";
        return;
    }
    vector<double> temp;
    string num;
    
    string line;
    getline(fin, line);
    while(getline(fin, line)) {
        temp.clear(), num.clear();
        for (int i = int(line.find(",")+1); i < line.length(); i++) {
            if (line[i] == ',') {
                double num1 = atof(num.c_str());
                temp.push_back(num1);
                num.clear();
            } else num += line[i];
        }
        
        training_set.push_back(temp);
        double num1 = atof(num.c_str());
        ref.push_back(num1);
        ref2.push_back(num1);
    }
    fin.close();
}

//生成标准正态分布的随机数
double sampleNormal() {
    //srand((unsigned int)time(0));
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return (u * c);
}