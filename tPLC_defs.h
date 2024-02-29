
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <vector>
#include <string>
#include <cmath>

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>


#define TPLC_BLOCK_LEN               320

#define TPLC_BLOCK_SHIFT             160  

#define PI 3.141592653589793238

#define MODEL_NAME "tPLCnet.tflite"

struct plc_engine {
    float audio_buffer[TPLC_BLOCK_LEN] = { 0 };
    float in_buffer[6*TPLC_BLOCK_SHIFT] = { 0 };
    float out_buffer[TPLC_BLOCK_LEN] = { 0 };

    TfLiteTensor* input_details[1];
    const TfLiteTensor* output_details[1];
    TfLiteInterpreter* interpreter;
    TfLiteModel* model;
};





#endif 



