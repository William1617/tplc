
 
#include "tPLC_defs.h"
#include "AudioFile.h"


void ExportWAV(
        const std::string & Filename, 
		const std::vector<float>& Data, 
		unsigned SampleRate) {
    AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);		
}

void TPLC() {

    trg_engine* m_pEngine;

    m_pEngine = new trg_engine;

	// load model
	m_pEngine->model = TfLiteModelCreateFromFile(MODEL_NAME);

    // Build the interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    // Create the interpreter.
    m_pEngine->interpreter = TfLiteInterpreterCreate(m_pEngine->model_a, options);
    if (m_pEngine->interpreter == nullptr) {
        printf("Failed to create interpreter\n");
        return ;
    }

    // Allocate tensor buffers.
    if (TfLiteInterpreterAllocateTensors(m_pEngine->interpreter) != kTfLiteOk) {
        printf("Failed to allocate tensors a!\n");
        return;}
    //Set input and output
    
    m_pEngine->input_details[0] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter, 0);
    m_pEngine->output_details[0] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter, 0);

    
    float windows[TPLC_BLOCK_LEN]={0};

    for (int i=0;i<TPLC_BLOCK_LEN;i++){
        m_windows[i]=0.5-0.5*cosf((2*PI*i)/(TPLC_BLOCK_LEN-1));
    }
    //Get loss id to each frame
    std::vector<int> loss_ids;
    std::string txtpath="./54lost.txt";
    std::fstream token_file;

    std::string tp;

    token_file.open(txtpath,std::ios::in);
    
    while(getline(token_file,tp)){
		loss_ids.push_back(stoi(tp));}

    token_file.close();


    std::vector<float>  testdata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputmicfile;
    std::string micfile="./54.wav";
    inputmicfile.load(micfile);
    
    int process_num=loss_ids.size();
    
    //for BLOCK_LEN input samples,do process_num infer
    for(int i=0;i<process_num;i++){
        memmove(m_pEngine->audio_buffer, m_pEngine->audio_buffer + TPLC_BLOCK_SHIFT, (TPLC_BLOCK_LEN -TPLC_BLOCK_SHIFT) * sizeof(float));
        for(int n=0;n<TPLC_BLOCK_SHIFT;n++){
                m_pEngine->audio_buffer[n+TPLC_BLOCK_LEN-TPLC_BLOCK_SHIFT]=inputmicfile.samples[0][n+i*TPLC_BLOCK_SHIFT];}

        memmove(m_pEngine->in_buffer,m_pEngine->in_buffer+TPLC_BLOCK_SHIFT,5*TPLC_BLOCK_SHIFT*sizeof(float));
        memcpy(m_pEngine->in_buffer+3*TPLC_BLOCK_SHIFT,m_pEngine->out_buffer,TPLC_BLOCK_SHIFT*sizeof(float));
        memcpy(m_pEngine->in_buffer+4*TPLC_BLOCK_SHIFT,m_pEngine->audio_buffer,2*TPLC_BLOCK_SHIFT*sizeof(float));
        bool is_loss=loss_ids[i]>0.5;
        if(is_loss){
            TPLCInfer(m_pEngine);

        }else{
            memmove(m_pEngine->out_buffer, m_pEngine->out_buffer + TPLC_BLOCK_SHIFT, (TPLC_BLOCK_LEN - TPLC_BLOCK_SHIFT) * sizeof(float));
            memset(m_pEngine->out_buffer + (TPLC_BLOCK_LEN - TPLC_BLOCK_SHIFT), 0, TPLC_BLOCK_SHIFT * sizeof(float));
            for (int i=0;i<TPLC_BLOCK_LEN;i++){
                m_pEngine->out_buffer[i] +=m_pEngine->audio_buffer[i]*m_windows[i];}
        }
        
        for(int j=0;j<TPLC_BLOCK_SHIFT;j++){
            testdata.push_back(m_pEngine->out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }

    ExportWAV("plctest.wav",testdata,SAMEPLERATE);
    TfLiteInterpreterDelete(m_pEngine->interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(m_pEngine->model);

 }
 
void TPLCInfer(trg_engine* m_pEngine) {


    TfLiteTensorCopyFromBuffer(m_pEngine->input_details[0], m_pEngine->in_buffer, 6*TPLC_BLOCK_SHIFT * sizeof(float));

    if (TfLiteInterpreterInvoke(m_pEngine->interpreter) != kTfLiteOk) {
        printf("Error invoking detection model\n");
        return;
    }

    float out_block[TPLC_BLOCK_LEN];
    TfLiteTensorCopyToBuffer(m_pEngine->output_details[0],out_block, TPLC_BLOCK_LEN * sizeof(float));
    

    //apply overlap_add
    memmove(m_pEngine->out_buffer, m_pEngine->out_buffer + TPLC_BLOCK_SHIFT, (TPLC_BLOCK_LEN - TPLC_BLOCK_SHIFT) * sizeof(float));
    memset(m_pEngine->out_buffer + (TPLC_BLOCK_LEN - TPLC_BLOCK_SHIFT), 0, TPLC_BLOCK_SHIFT * sizeof(float));
    for (int i = 0; i < TPLC_BLOCK_LEN; i++)
        m_pEngine->out_buffer[i] += out_block[i];

}
 



