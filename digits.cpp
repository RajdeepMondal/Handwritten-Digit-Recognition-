#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
//sigmoid function
float sigmoid(float z){
	return (1.0/(1.0+exp(-z)));
}
//feedforward propagation algorithm
void feedforward(float theta1[][401],float theta2[][26],float a1[],float a2[],float a3[]){
	int i,j;
	a2[0]=1;
	for(i=1;i<26;i++){
		a2[i]=0;
		for(j=0;j<401;j++) a2[i]+=theta1[i-1][j]*a1[j]; 
	}
	for(i=1;i<26;i++) a2[i]=sigmoid(a2[i]);
	for(i=0;i<10;i++){
		a3[i]=0;
		for(j=0;j<26;j++) a3[i]+=theta2[i][j]*a2[j];
	}
	for(i=0;i<10;i++) a3[i]=sigmoid(a3[i]);
}
//Computing cost function J(theta)
float cost_function(int y[][10],float **X_train,float theta1[][401],float theta2[][26],float a1[],float a2[],float a3[]){
	int i,j,k;
	float cost=0;
	for(i=0;i<5000;i++){
		for(j=0;j<401;j++) a1[j]=X_train[i][j];
		feedforward(theta1,theta2,a1,a2,a3);
		for(k=0;k<10;k++){
			cost+=-(y[i][k]*log(a3[k])+(1-y[i][k])*log(1-a3[k]));
		}
	}
	cost/=5000;
	return cost;
}
//backpropagation algorithm
void backpropagation(float theta2[][26],float a2[],float a3[],int ind,int y[][10],float delta2[],float delta3[]){
	int i,j;
	float theta_t[26][10];
	for(i=0;i<10;i++) delta3[i]=a3[i]-y[ind][i];
	for(i=0;i<26;i++){
		for(j=0;j<10;j++){
			theta_t[i][j]=theta2[j][i];
		}
	}
	for(i=0;i<26;i++){
		delta2[i]=0;
		for(j=0;j<10;j++){
			delta2[i]+=theta_t[i][j]*delta3[j];
		}
		delta2[i]*=a2[i]*(1-a2[i]);
	}
}
//gradient calculation
void grad(float theta1[][401],float theta2[][26],float a1[],float a2[],float a3[],float **X_train,int y[][10],float D1[][401],float D2[][26]){
	int i,j,k;
	float delta2[26],delta3[10];
	for(i=0;i<25;i++){
		for(j=0;j<401;j++){
			D1[i][j]=0;
		}
	}
	for(i=0;i<10;i++){
		for(j=0;j<26;j++){
			D2[i][j]=0;
		}
	}
	for(i=0;i<5000;i++){
		for(j=0;j<401;j++) a1[j]=X_train[i][j];
		feedforward(theta1,theta2,a1,a2,a3);
		backpropagation(theta2,a2,a3,i,y,delta2,delta3);
		for(j=0;j<25;j++){                      //
			for(k=0;k<401;k++){                 //
				D1[j][k]+=delta2[j+1]*a1[k];   //
			}
		}
		for(j=0;j<10;j++){
			for(k=0;k<26;k++){
				D2[j][k]+=delta3[j]*a2[k];
			}
		}
	}
	for(j=0;j<25;j++){
		for(k=0;k<401;k++) D1[j][k]/=5000;
	}
	for(j=0;j<10;j++){
		for(k=0;k<26;k++) D2[j][k]/=5000;
	}
	
}
//batch gradient descent
void gradient_descent(float theta1[][401],float theta2[][26],float **X_train,int y[][10],float a1[],float a2[],float a3[]){
	int i,j,k;
	float D1[25][401],D2[10][26];
	for(k=0;k<10000;k++){
		printf("%f\n",cost_function(y,X_train,theta1,theta2,a1,a2,a3));
		grad(theta1,theta2,a1,a2,a3,X_train,y,D1,D2);	
		for(i=0;i<25;i++){
			for(j=0;j<401;j++){
				theta1[i][j]-=0.1*D1[i][j];
			}
		}
		for(i=0;i<10;i++){
			for(j=0;j<26;j++){
				theta2[i][j]-=0.1*D2[i][j];
			}
		}
	}
}
int compare(float a3[]){
	int i,max=0;
	float m=a3[0];
	for(i=1;i<10;i++){
		if(a3[i]>m){
			m=a3[i];
			max=i;
		}
	}
	return max;
}
int main(){
	/*
	theta1->parameters of layer 1 to layer 2 ----- dimension 25 x 401
	theta2->parameters of layer 2 to layer 3 ----- dimension 10 x 26
	a1->activation of layer 1 ----- dimension 401 x 1 ----- input layer
	a2->activation of layer 2 ----- dimension 26 x 1 ----- hidden layer
	a3->activation of layer 3 ----- dimension 10 x 1 ----- output layer
	*/
	float **X,**X_train,theta1[25][401],theta2[10][26],a1[401],a2[26],a3[10];
	int i,j,y[5000][10],d;
	FILE *inpf;
	X=(float**)malloc(5000*sizeof(float*));
	X_train=(float**)malloc(5000*sizeof(float*));
	//y=(int**)malloc(5000*sizeof(int*));
	for(i=0;i<5000;i++) {
		X[i]=(float*)malloc(400*sizeof(float));
		//y[i]=(int*)malloc(10*sizeof(int));
		X_train[i]=(float*)malloc(401*sizeof(float));
	}
	inpf=fopen("input.txt","r");
	if(inpf==NULL){
		printf("File not found\n");
		exit(-1);
	}
	for(i=0;i<5000;i++){
		for(j=0;j<400;j++) fscanf(inpf,"%f",&X[i][j]);
	}
	for(i=0;i<5000;i++){
		fscanf(inpf,"%d",&d);
		for(j=0;j<10;j++) y[i][j]=0;
		for(j=0;j<10;j++){
			if(d==10) y[i][0]=1;
			else y[i][d]=1;
		}
	}
	for(i=0;i<5000;i++){
		X_train[i][0]=1;
		for(j=1;j<=400;j++) X_train[i][j]=X[i][j-1];
	}
	for(i=0;i<25;i++){
		for(j=0;j<401;j++) theta1[i][j]=((float)rand())/RAND_MAX - 0.5;
	}
	for(i=0;i<10;i++){
		for(j=0;j<26;j++) theta2[i][j]=((float)rand())/RAND_MAX - 0.5;
	}
	gradient_descent(theta1,theta2, X_train,y,a1,a2,a3);
	int count=0,out;
	for(i=0;i<5000;i++){
		for(j=0;j<401;j++) a1[j]=X_train[i][j];
		feedforward(theta1,theta2,a1,a2,a3);
		out=compare(a3);
		if(y[i][out]==1) count++;
	}
	printf("\n\n\naccuracy: %f\n",(float)count*100/5000);
	return 0;
}
