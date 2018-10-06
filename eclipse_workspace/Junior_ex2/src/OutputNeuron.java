public class OutputNeuron extends Neuron{
	//member field

	//member method
	public void reWeight(double y, double o_fromOutput, InterNeuron inter[]){
		for(int i=0; i<weight.length; i++){
			delta_W[i] *= alpha;
			delta_W[i] += eta * (y - o_fromOutput) * o_fromOutput * (1 - o_fromOutput) * inter[i].output();
			weight[i] += delta_W[i];	//重み更新式
		}
		delta_T *= alpha;
		delta_T += eta * (y - o_fromOutput) * o_fromOutput * (1 - o_fromOutput);
		threshoud += delta_T;	//しきい値更新式
	}

	//constructor
	OutputNeuron(int interNumber, double preWeight, double preThreshoud, double preEta, double preAlpha) {
		super(interNumber, preWeight, preThreshoud, preEta, preAlpha);
	}
}
