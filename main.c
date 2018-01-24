#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <memory.h>
#include <wait.h>
#include <pthread.h>

#define GENERATION 500
#define PROBABILITY_CROSSOVER 0.7
#define PROBABILITY_TORNAMENT 0.7
#define ROYAL_CONSTANT 1
#define FIRST_INDIVIDUAL 200
#define FIRST_NUM_OF_ELITE 4
#define FIRST_PROBABILITY_MUTATION 0.2

int INDIVIDUAL = FIRST_INDIVIDUAL;
double PROBABILITY_MUTATION = FIRST_PROBABILITY_MUTATION;
int NUM_OF_ELITE = FIRST_NUM_OF_ELITE;
double sum;

const int GENES = 16;
const float background_volume_max = 0.4;
const float background_volume_min = 0.05;
const float background_frequency_max = 1;
const float background_frequency_min = 0.2;
const float silence_percentage_max = 40;
const float silence_percentage_min = 5;
const float unknown_percentage_max = 40;
const float unknown_percentage_min = 5;
const float time_shift_ms_max = 400;
const float time_shift_ms_min = 50;
const int window_size_ms_max = 120;
const int window_size_ms_min = 15;
const int window_stride_ms_max = 40;
const int window_stride_ms_min = 5;
const int dct_coefficient_count_max = 160;
const int dct_coefficient_count_min = 20;

//float window_stride_ms = 0;
const int how_many_training_steps_max = 100;

const float dropout_prob_max = 0.5;
const float dropout_prob_min = 0;
const int filter_width_max = 32;
const int filter_width_min = 4;
const int filter_height_max = 80;
const int filter_height_min = 10;
const int filter_count_max = 256;
const int filter_count_min = 32;
const int max_pool_min = 2;

int fitValue;
typedef struct parameter
{
	 float background_volume;
	 float background_frequency;
	 float silence_percentage;
	 float unknown_percentage;
	 float time_shift_ms;
	 float window_size_ms;
	 float window_stride_ms;
	 int dct_coefficient_count;
	 int how_many_training_steps[2];

	 float First_dropout_prob;
	 int First_filter_width;//2번째 인자
	 int First_filter_height;//3번째 인자
	 int First_filter_count;//4번째 인자
	 int First_convolution_Xavier_or_not; //0일때는 no 1일때는 Xavier
	 int First_max_pool_or_not;
	 int First_max_pool[2];//max_pool[0] = width
	 int First_max_pool_padding;//padding = 1 means valid
	 int First_convolution_padding;
	 
	 float Second_dropout_prob;
	 int Second_filter_width;
	 int Second_filter_height;
	 int Second_filter_count;
	 int Second_max_pool_or_not;
	 int Second_max_pool[2];//max_pool[0] = width
	 int Second_convolution_Xavier_or_not;//0일때는 no 1일때는 yes
	 int Second_max_pool_padding;//padding = 1 means valid
	 int Second_convolution_padding;

	 float Third_dropout_prob;
	 int Third_filter_width;
	 int Third_filter_height;
	 int Third_filter_count;
	 int Third_convolution_Xavier_or_not;//0일때는 no 1일때는 yes
	 int Third_convolution_padding;
	 int fit;
}Parameter;

typedef struct fitness
{
	double ideal;
	double average;
	int indexOfIdeal;
	double fit[FIRST_INDIVIDUAL];
}Fitness;

Parameter *population;
Parameter *next_population;
Fitness *generation;

void convolution(int height, int width, int* max_pool, int padding, int* filter)
{
	int newWidth = width / max_pool[0];
	int newHeight = height / max_pool[1];

	if(padding == 0)
	{
		if(width * max_pool[0] != newWidth)
			newWidth += 1;
		if(height * max_pool[1] != newHeight)
			newHeight +=1;
	}

	filter[0] = newWidth;
	filter[1] = newHeight;
}

void update_convolution_paramter(Parameter* parameter)
{
	int* filter = malloc(sizeof(int) * 2);
	
	int length_minus_window = 16000 - 16 * parameter->window_size_ms;
	int window_stride_sample = 16 * parameter->window_stride_ms;
	int width_max = 1 + length_minus_window / window_stride_sample;
	if(width_max > filter_width_max)
		width_max = filter_width_max;
	parameter->First_filter_width = rand() % (width_max - filter_width_min) + filter_width_min;

	int height_max = 40;
	if(height_max > filter_height_max)
		height_max = filter_height_max;
	parameter->First_filter_height = rand()%(height_max - filter_height_min) + filter_height_min;

	parameter->First_filter_count = rand() % (filter_count_max- filter_count_min) + filter_count_min;
	parameter->First_dropout_prob = ((float)rand() / (float)RAND_MAX) *(dropout_prob_max - dropout_prob_min);
	parameter->First_convolution_Xavier_or_not = rand() % 2;

	parameter->First_max_pool[0] = (rand()% (parameter->First_filter_width))/2 +1;
	if(parameter->First_max_pool[0]> 8)
		parameter->First_max_pool[0] = 8;

	parameter->First_max_pool[1] = (rand()% (parameter->First_filter_height))/2 +1;
	if(parameter->First_max_pool[1] > 8)
		parameter->First_max_pool[1] = 8;

	parameter->First_max_pool_padding= rand()%2;
	parameter->First_convolution_padding = rand()%2;
	parameter->First_max_pool_or_not = rand()%2;
	filter[0] = parameter->First_filter_width;
	filter[1] = parameter->First_filter_height;

	if(parameter->First_max_pool_or_not== 1)
	{
		convolution(parameter->First_filter_height, parameter->First_filter_width,
				parameter->First_max_pool, parameter->First_max_pool_padding, filter);
	}
	parameter->Second_filter_width = filter[0];
	parameter->Second_filter_height = filter[1];

	parameter->Second_filter_count = rand() % 224 + 32;
	parameter->Second_dropout_prob = ((float)rand() / (float)RAND_MAX) *(dropout_prob_max - dropout_prob_min);
	parameter->Second_convolution_Xavier_or_not = rand()%2;

	parameter->Second_filter_count = rand() % (filter_count_max- filter_count_min) + filter_count_min;
	parameter->Second_dropout_prob = ((float)rand() / (float)RAND_MAX) *(dropout_prob_max - dropout_prob_min);
	parameter->Second_convolution_Xavier_or_not = rand() % 2;

	parameter->Second_max_pool[0] = (rand()% (parameter->Second_filter_width))/2 +1;
	if(parameter->Second_max_pool[0]> 8)
		parameter->Second_max_pool[0] = 8;

	parameter->Second_max_pool[1] = (rand()% (parameter->Second_filter_height))/2 +1;
	if(parameter->Second_max_pool[1] > 8)
		parameter->Second_max_pool[1] = 8;

	parameter->Second_convolution_padding = rand()%2;
	parameter->Second_max_pool_padding = rand()%2;
	parameter->Second_max_pool_or_not = rand()%2;

	if(parameter->Second_max_pool_or_not== 1)
	{
		convolution(parameter->Second_filter_height, parameter->Second_filter_width,
				parameter->Second_max_pool, parameter->Second_max_pool_padding, filter);
	}
	parameter->Third_filter_width = filter[0];
	parameter->Third_filter_height = filter[1];
	parameter->Third_dropout_prob = ((float)rand() / (float)RAND_MAX) *(dropout_prob_max - dropout_prob_min);
	parameter->Third_convolution_Xavier_or_not = rand() % 2;
	parameter->Third_filter_count = rand() % (filter_count_max- filter_count_min) + filter_count_min;
	parameter->Third_convolution_padding = rand() % 2;
	parameter->Third_convolution_padding = rand() % 2;
	

	free(filter);
}

void initialize()
{
	 for (int i = 0; i < INDIVIDUAL; i++)
	 {
		  population[i].background_volume = ((float)rand() / (float)RAND_MAX)* (background_volume_max - background_volume_min) + background_volume_min;
		  population[i].background_frequency = ((float)rand() / (float)RAND_MAX)*(background_frequency_max - background_frequency_min) + background_frequency_min;
		  population[i].silence_percentage = rand() % ((int)silence_percentage_max - (int)silence_percentage_min) + silence_percentage_min;
		  population[i].unknown_percentage = rand() % ((int)unknown_percentage_max - (int)unknown_percentage_min) + unknown_percentage_min;
		  population[i].time_shift_ms = rand() % ((int)time_shift_ms_max - (int)time_shift_ms_min) + time_shift_ms_min;
		  population[i].window_size_ms = rand() % ((int)window_size_ms_max-window_size_ms_min) +window_size_ms_min;//Fiilt_filter_width 에서 나누기하기 때문에 +1 을
		  int tempMax;
		  if(window_stride_ms_max > population[i].window_size_ms)
			  tempMax = population[i].window_size_ms;
		  else
			  tempMax = window_stride_ms_max;
		  population[i].window_stride_ms = rand() % (tempMax - window_stride_ms_min) + 1;
		  population[i].dct_coefficient_count = (rand() % dct_coefficient_count_max - dct_coefficient_count_min)+dct_coefficient_count_min;//First_filter_height에서 나누기 하기때문에 여기도 +1을 해줌
		  population[i].how_many_training_steps[0] = rand() % how_many_training_steps_max;
		  population[i].how_many_training_steps[1] = how_many_training_steps_max - population[i].how_many_training_steps[0];
		  population[i].fit = 0 ;
		  update_convolution_paramter(&population[i]);
	 }
}

int tornament(Fitness* generation)
{
	int rand1, rand2;
	double prob;

	rand1 = rand()%INDIVIDUAL;
	rand2 = rand()%INDIVIDUAL;
	prob = ((double)rand()/(RAND_MAX));

	if(prob<PROBABILITY_TORNAMENT)
	{
		if(generation->fit[rand1]> generation->fit[rand2])
			return rand1;
		else
			return rand2;
	}
	else
	{
		if(generation->fit[rand1]> generation->fit[rand2])
			return rand2;
		else
			return rand1;
	}
}

void crossover(Parameter* chromosome1, Parameter* chromosome2, int index1, int index2)
{
	int gene = (int)rand() % GENES;

	*chromosome1 = population[index2];
	*chromosome2 = population[index1];

	memcpy(chromosome1, &population[index1], (gene+1)*4);
	memcpy(chromosome2, &population[index2], (gene+1)*4);
}

void one_child_crossover(Parameter* chromosome1, int index1, int index2)
{
	int gene = (int)rand() % GENES;

	*chromosome1 = population[index2];

	memcpy(chromosome1, &population[index1], (gene+1)*4);
}

void elitism(Parameter* elite, int generationValue)
{
	for(int i = 0 ; i < NUM_OF_ELITE;i++)
		population[i] = elite[i];
	FILE *f;
	char fileName[32];
	sprintf(fileName, "ELITE/GENE%d.txt", generationValue);
	f = fopen(fileName, "w+");
	fclose(f);
	f = fopen(fileName, "a+");
	for(int idex = 0 ; idex < NUM_OF_ELITE ; idex++)
	{
	fprintf(f, "%f %f %f %f %f %f %f %d %d %d %f %d %d %d %d %d %d %d %d %d %f %d %d %d %d %d %d %d %d %d %f %d %d %d %d %d %d\n",
	population[idex].background_volume,population[idex].background_frequency,population[idex].silence_percentage,population[idex].unknown_percentage,population[idex].time_shift_ms,
	population[idex].window_size_ms,population[idex].window_stride_ms,population[idex].dct_coefficient_count,population[idex].how_many_training_steps[0],population[idex].how_many_training_steps[1],
	population[idex].First_dropout_prob,population[idex].First_filter_width,population[idex].First_filter_height,population[idex].First_filter_count,population[idex].First_convolution_Xavier_or_not,
	population[idex].First_max_pool_or_not,population[idex].First_max_pool[0],population[idex].First_max_pool[1],population[idex].First_max_pool_padding,population[idex].First_convolution_padding,
	population[idex].Second_dropout_prob,population[idex].Second_filter_width,population[idex].Second_filter_height,population[idex].Second_filter_count,population[idex].Second_max_pool_or_not,
	population[idex].Second_max_pool[0],population[idex].Second_max_pool[1],population[idex].Second_convolution_Xavier_or_not,population[idex].Second_max_pool_padding,
	population[idex].Second_convolution_padding,population[idex].Third_dropout_prob,population[idex].Third_filter_width,population[idex].Third_filter_height,population[idex].Third_filter_count,
	population[idex].Third_convolution_Xavier_or_not,population[idex].Third_convolution_padding, population[idex].fit);
	}
	fclose(f);
}

void pySystemCall(char* str)
{
	fitValue = system(str);
	fitValue %= 255;
	pthread_exit(NULL);
}

double fitness(int idex, int generationValue)
{
	char str[10240];
	char nullStr[100] = "python3 train.py";
	char temp[500];
	strcpy(str,nullStr);

	sprintf(temp," --background_volume %f", population[idex].background_volume);
	strcat(str,temp);
	sprintf(temp," --background_frequency %f", population[idex].background_frequency);
	strcat(str,temp);
	sprintf(temp," --silence_percentage %f", population[idex].silence_percentage);
	strcat(str,temp);
	sprintf(temp," --unknown_percentage %f", population[idex].unknown_percentage);
	strcat(str,temp);
	sprintf(temp," --time_shift_ms %f", population[idex].time_shift_ms);//5
	strcat(str,temp);
	sprintf(temp," --window_size_ms %f", population[idex].window_size_ms);
	strcat(str,temp);
	sprintf(temp," --window_stride_ms %f", population[idex].window_stride_ms);
	strcat(str,temp);
	sprintf(temp," --how_many_training_steps %d,%d",
			population[idex].how_many_training_steps[0],population[idex].how_many_training_steps[1]);
	strcat(str,temp);

	sprintf(temp, " --first_dropout_prob %f", population[idex].First_dropout_prob);
	strcat(str,temp);
	sprintf(temp, " --first_filter_width %d", population[idex].First_filter_width);
	strcat(str,temp);
	sprintf(temp, " --first_filter_height %d", population[idex].First_filter_height);
	strcat(str,temp);
	sprintf(temp, " --first_filter_count %d", population[idex].First_filter_count);
	strcat(str,temp);
	sprintf(temp, " --first_convolution_xavier_or_not %d", population[idex].First_convolution_Xavier_or_not);//15
	strcat(str,temp);
	sprintf(temp, " --first_convolution_stride %d %d", 1,1);
	strcat(str,temp);
	sprintf(temp, " --first_convolution_padding %d", population[idex].First_convolution_padding);
	strcat(str,temp);
	sprintf(temp, " --first_maxpool_or_not %d", population[idex].First_max_pool_or_not);
	strcat(str,temp);
	sprintf(temp, " --first_maxpool_stride %d %d",
			population[idex].First_max_pool[0], population[idex].First_max_pool[1]);
	strcat(str,temp);
	sprintf(temp, " --first_maxpool_padding %d",population[idex].First_max_pool_padding);
	strcat(str,temp);

	sprintf(temp, " --second_dropout_prob %f", population[idex].Second_dropout_prob);//20
	strcat(str,temp);
	sprintf(temp, " --second_filter_width %d", population[idex].Second_filter_width);
	strcat(str,temp);
	sprintf(temp, " --second_filter_height %d", population[idex].Second_filter_height);
	strcat(str,temp);
	sprintf(temp, " --second_filter_count %d", population[idex].Second_filter_count);
	strcat(str,temp);
	sprintf(temp, " --second_convolution_xavier_or_not %d",
			population[idex].Second_convolution_Xavier_or_not);
	strcat(str,temp);
	sprintf(temp, " --second_convolution_stride %d %d", 1,1);
	strcat(str, temp);
	sprintf(temp, " --second_convolution_padding %d", population[idex].Second_convolution_padding);
	strcat(str, temp);
	sprintf(temp, " --second_maxpool_or_not %d", population[idex].Second_max_pool_or_not);//25
	strcat(str,temp);
	sprintf(temp, " --second_maxpool_stride %d %d",
			population[idex].Second_max_pool[0],population[idex].Second_max_pool[1]);
	strcat(str,temp);
	sprintf(temp, " --second_maxpool_padding %d", population[idex].Second_max_pool_padding);
	strcat(str,temp);
	sprintf(temp, " --third_dropout_prob %f", population[idex].Third_dropout_prob);
	strcat(str,temp);
	sprintf(temp, " --third_filter_width %d", population[idex].Third_filter_width);//30
	strcat(str,temp);
	sprintf(temp, " --third_filter_height %d", population[idex].Third_filter_height);
	strcat(str,temp);
	sprintf(temp, " --third_filter_count %d", population[idex].Third_filter_count);
	strcat(str,temp);
	sprintf(temp, " --third_convolution_xavier_or_not %d", population[idex].Third_convolution_Xavier_or_not);//33
	strcat(str,temp);
	sprintf(temp, " --third_convolution_padding %d",population[idex].Third_convolution_padding);
	strcat(str,temp);
	sprintf(temp, " --third_convolution_stride %d %d", 1,1);
	strcat(str,temp);

	int num;

	int length_minus_window = 16000 - 16 * population[idex].window_size_ms;
	int window_stride_sample = 16 * population[idex].window_stride_ms;
	int width_max =1 + length_minus_window/ window_stride_sample;
	int height_max =  40;

	printf("\n\n\n\n%d 번째 실행중...\n\n\n\n", idex);

	char systemStr[64];
	fitValue = 0;
	int nWorkStatus = 0;
	int thread_id;
	pthread_t pthread;
	int status;

	thread_id = pthread_create(&pthread, NULL, pySystemCall, (char*)str);
	
	pthread_join(pthread, &status);

	population[idex].fit = fitValue;
	saveGA(idex, generationValue);
	
	return fitValue;
}

void saveGA(int number, int generationValue)
{
	FILE *f;
	char fileName[32];
	sprintf(fileName, "population/set%d.txt", number/10);
	f = fopen(fileName, "w+");
	fclose(f);
	f = fopen(fileName, "a+");
	for(int idex = (number/10) * 10 ; idex < (number/10)*10 +10 ; idex ++)
	{
		fprintf(f, "%f %f %f %f %f %f %f %d %d %d %f %d %d %d %d %d %d %d %d %d %f %d %d %d %d %d %d %d %d %d %f %d %d %d %d %d %d\n",
		population[idex].background_volume,population[idex].background_frequency,population[idex].silence_percentage,population[idex].unknown_percentage,population[idex].time_shift_ms,
		population[idex].window_size_ms,population[idex].window_stride_ms,population[idex].dct_coefficient_count,population[idex].how_many_training_steps[0],population[idex].how_many_training_steps[1],
		population[idex].First_dropout_prob,population[idex].First_filter_width,population[idex].First_filter_height,population[idex].First_filter_count,population[idex].First_convolution_Xavier_or_not,
		population[idex].First_max_pool_or_not,population[idex].First_max_pool[0],population[idex].First_max_pool[1],population[idex].First_max_pool_padding,population[idex].First_convolution_padding,
		population[idex].Second_dropout_prob,population[idex].Second_filter_width,population[idex].Second_filter_height,population[idex].Second_filter_count,population[idex].Second_max_pool_or_not,
		population[idex].Second_max_pool[0],population[idex].Second_max_pool[1],population[idex].Second_convolution_Xavier_or_not,population[idex].Second_max_pool_padding,
		population[idex].Second_convolution_padding,population[idex].Third_dropout_prob,population[idex].Third_filter_width,population[idex].Third_filter_height,population[idex].Third_filter_count,
		population[idex].Third_convolution_Xavier_or_not,population[idex].Third_convolution_padding, population[idex].fit);
	}
	fclose(f);
	f = fopen("information.txt", "w+");
	fprintf(f, "%d %d %lf %d", generationValue, number, PROBABILITY_MUTATION, NUM_OF_ELITE);
	fclose(f);
}

void loadGA(int number, int generationValue)
{
	FILE *f;
	char filename[32];
	sprintf(filename, "population/set%d.txt", number/10);
	f = fopen(filename, "r+");
		
	for(int idex = (number /10) * 10 ; idex < (number/10)*10 +10 ; idex ++)
	{
		fscanf(f, "%f %f %f %f %f %f %f %d %d %d %f %d %d %d %d %d %d %d %d %d %f %d %d %d %d %d %d %d %d %d %f %d %d %d %d %d %d",
		&population[idex].background_volume,&population[idex].background_frequency,&population[idex].silence_percentage,&population[idex].unknown_percentage,&population[idex].time_shift_ms,
		&population[idex].window_size_ms,&population[idex].window_stride_ms,&population[idex].dct_coefficient_count,&population[idex].how_many_training_steps[0],&population[idex].how_many_training_steps[1],
		&population[idex].First_dropout_prob,&population[idex].First_filter_width,&population[idex].First_filter_height,&population[idex].First_filter_count,&population[idex].First_convolution_Xavier_or_not,
		&population[idex].First_max_pool_or_not,&population[idex].First_max_pool[0],&population[idex].First_max_pool[1],&population[idex].First_max_pool_padding,&population[idex].First_convolution_padding,
		&population[idex].Second_dropout_prob,&population[idex].Second_filter_width,&population[idex].Second_filter_height,&population[idex].Second_filter_count,&population[idex].Second_max_pool_or_not,
		&population[idex].Second_max_pool[0],&population[idex].Second_max_pool[1],&population[idex].Second_convolution_Xavier_or_not,&population[idex].Second_max_pool_padding,
		&population[idex].Second_convolution_padding,&population[idex].Third_dropout_prob,&population[idex].Third_filter_width,&population[idex].Third_filter_height,&population[idex].Third_filter_count,
		&population[idex].Third_convolution_Xavier_or_not,&population[idex].Third_convolution_padding, &population[idex].fit);
		generation[generationValue].fit[idex] = population[idex].fit;
	}
	fclose(f);
}

void initNew()
{
	for(int i = 1 ; i < 20; i ++)
		saveGA(i*10,0);
	saveGA(0,0);

	FILE *f;
	char fileName[32];
	sprintf(fileName, "fitness/gene%d.txt", 0);
	f = fopen(fileName,"w");
	sum = 0;
	generation[0].ideal = 0;
	generation[0].indexOfIdeal = 0;
	fprintf(f, "%lf %lf %d", 0,0,0);
	fclose(f);
}

void initLoad(int* generationValue, int* indiviValue)
{
	FILE *f;
	f = fopen("information.txt", "r");
	fscanf(f, "%d %d %lf %d", generationValue, indiviValue, &PROBABILITY_MUTATION, &NUM_OF_ELITE);
	fclose;

	for(int i = 0 ; i < 20 ; i ++)
	{
		loadGA(i*10, generationValue);
	}

	for(int i = 0 ; i < *generationValue; i ++)
	{
		FILE *f;
		char filename[32];
		sprintf(filename, "fitness/gene%d.txt", i);
		f = fopen(filename, "r");
		int idea;
		int indexOfIdeal;
		fscanf(f, "%lf %lf %d", &idea, &sum, &indexOfIdeal); 
		generation[i].ideal = idea;
		generation[i].indexOfIdeal = indexOfIdeal;
		generation[i].average = sum / INDIVIDUAL;
		fclose(f);
	}
}


void fitnessCheck(Fitness* generation, int generationValue, int indiviValue)
{
	double ideal = 0;
	int indexOfIdeal = 0;
	double sum = 0;
	int i = 0;

	for(i = indiviValue ; i < INDIVIDUAL ; i ++)
	{
		generation->fit[i] = fitness(i, generationValue);
		sum += generation->fit[i];

		if( generation->fit[i] > ideal)
		{
			ideal = generation->fit[i];
			indexOfIdeal = i;
		}
		FILE *f;
		char filename[32];
		sprintf(filename, "fitness/gene%d.txt", generationValue);
		f = fopen(filename, "w");
		fprintf(f, "%lf %lf %d", ideal, sum, indexOfIdeal);
		fclose(f);
	}
	generation->ideal = ideal;
	generation->average=sum/INDIVIDUAL;
	generation->indexOfIdeal=indexOfIdeal;
}

int main(int argc, char* argv[])
{
	struct timeval t;
	int i, j;
	gettimeofday(&t, NULL);
	srand(t.tv_usec);

	population = (Parameter*)malloc(sizeof(Parameter) * INDIVIDUAL);
	next_population = (Parameter*)malloc(sizeof(Parameter) * INDIVIDUAL);
 	generation = (Fitness*)malloc(sizeof(Fitness) * GENERATION);

	int indiviValue;
	if(strcmp(argv[1], "new") == 0)
	{
		initialize();
		i=0;
		initNew();
	}
	else if(strcmp(argv[1], "load") ==0)
	{
		initLoad(&i,&indiviValue);//i 가 뜻하는 것은 세대
	}

	 for (; i < GENERATION; i++)
	 {
		 fitnessCheck(&generation[i], i,indiviValue );

		 if((i+1)%125 == 0)
		 {
			 INDIVIDUAL /= 2;
			 if(i != 374 )
				 NUM_OF_ELITE /= 2;
			 if(i == 249)
			 {
				PROBABILITY_MUTATION /= 2;
			 }
		 }

		 int indices_selec[FIRST_INDIVIDUAL];

		 for(j = 0 ; j < INDIVIDUAL; j ++)
		 {
			 indices_selec[j] = tornament(&generation[i]);
		 }

		 for(j = 0 ; j < INDIVIDUAL; j++)
			 next_population[j] = population[j];

		 for(j = 0 ; j < INDIVIDUAL ; j +=2)
		 {
			 double prob = ((double) rand()/ (RAND_MAX));

			 if(prob > PROBABILITY_CROSSOVER)
				 continue;

			 Parameter chromosome1;

			 if((i+1)%125 == 0)
			 {
				 one_child_crossover(&chromosome1, indices_selec[j*2],indices_selec[j*2+1]);
				 next_population[j] = chromosome1;
			 }
			 else
			 {
				 Parameter chromosome2;
				 crossover(&chromosome1, &chromosome2, indices_selec[j], indices_selec[j+1]);

			 	 next_population[j] = chromosome1;
			 	 next_population[j+1] = chromosome2;
			 }
		 }

		 //crossover 수행
		 for(j = 0 ; j < INDIVIDUAL; j++)
		 {
			 double prob;
			 prob = (double) rand() / (RAND_MAX);
			 if(prob < PROBABILITY_MUTATION)
			 {
				  population[j].background_volume = ((float)rand() / (float)RAND_MAX)* (background_volume_max - background_volume_min) + background_volume_min;
				  population[j].background_frequency = ((float)rand() / (float)RAND_MAX)*(background_frequency_max - background_frequency_min) + background_frequency_min;
				  population[j].silence_percentage = rand() % ((int)silence_percentage_max - (int)silence_percentage_min) + silence_percentage_min;
				  population[j].unknown_percentage = rand() % ((int)unknown_percentage_max - (int)unknown_percentage_min) + unknown_percentage_min;
				  population[j].time_shift_ms = rand() % ((int)time_shift_ms_max - (int)time_shift_ms_min) + time_shift_ms_min;
				  population[j].window_size_ms = rand() % ((int)window_size_ms_max-window_size_ms_min) +window_size_ms_min;//Fiilt_filter_width 에서 나누기하기 때문에 +1 을
				  int tempMax;
				  if(window_stride_ms_max > population[j].window_size_ms)
					  tempMax = population[j].window_size_ms;
				  else
					  tempMax = window_stride_ms_max;
				  population[j].window_stride_ms = rand() % (tempMax - window_stride_ms_min) + 1;
				  population[j].dct_coefficient_count = (rand() % dct_coefficient_count_max - dct_coefficient_count_min)+dct_coefficient_count_min;//First_filter_height에서 나누기 하기때문에 여기도 +1을 해줌
				  population[j].how_many_training_steps[0] = rand() % how_many_training_steps_max;
				  population[j].how_many_training_steps[1] = how_many_training_steps_max - population[j].how_many_training_steps[0];
				  population[j].fit = 0;
				  update_convolution_paramter(&population[j]);
			 }
		 }

		 int tmp[INDIVIDUAL];
		 Parameter eliteGenes[FIRST_NUM_OF_ELITE];
		 int max=0;
		 int index=0;

		 for(int m = 0 ; m < INDIVIDUAL; m ++)
		 {
			 tmp[m] = generation[i].fit[m];
		 }

		 for(int idx = 0 ; idx < NUM_OF_ELITE ; idx++)
		 {
			 for(int m = 0 ; m < INDIVIDUAL; m ++)
			 {
				 if(tmp[m] > max)
				 {
					 max = tmp[m];
					 index = m;
				 }
			 }
			 tmp[index] = 0;
			 max = 0;

			 eliteGenes[idx] = population[index];
		 }

		 for(j = 0 ; j < INDIVIDUAL ; j++)
		 {
			 population[j] = next_population[j];
		 }
		 //replacement
		 elitism(eliteGenes, i);
		 //elitism
		 for(int ii = 1 ; ii < 20 ; ii++)
			 saveGA(ii*10,i);
		 saveGA(0,i);
	 }

	return 0;
}
