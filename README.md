## Online and Federated Learning for Predictive Maintenance

*In collaboration with Scania AB*

<div align="justify">
New heavy-duty vehicles have multiple actuators and sensors that are used to generate and record time-series data streams, respectively. These data streams can be used to model what normal operations in trucks are supposed to look like and, by extension, can also be used to identify anomalies. These anomalies could be engine overheating, malfunction in the ABS, or fatigue in suspensions. We implemented a transformer-based variational autoencoder [<a href="https://arxiv.org/pdf/1706.03762">1</a>] [<a href="https://arxiv.org/pdf/1312.6114">2</a>] to detect anomalies in truck operations for predictive maintenance. There are two reasons for selecting this model: 1) Because it has an explainable reconstruction loss, and 2) Because it can parallely process the entire time series window. Now we are faced with two challenges. Firstly, we want the model to generalize well to different operating conditions. For example, if there is a truck being operated in a city area and is then sent to remote hilly regions for long-haul missions, it is very likely that the underlying data distribution would change, and we expect the model to perform well in either scenario. Secondly, because trucks don't have large memory, we want a training regime that is memory efficient. Our solution: <i>online and federated learning</i>. Federated learning [<a href="https://arxiv.org/pdf/1602.05629">3</a>] is a collaborative way to jointly train multiple machine learning models without exposing the underlying data. It improves model generalizability while keeping the data private and reducing communication inefficiencies. As for memory limitations, we apply online learning where the model is trained on an incoming data sample (or a time window in our case), its parameters are updated based on the loss function, and then the sample is discarded. Thus, we don't need to store historical data as in batch training. <i>We primarily answer two questions: </i> 1) How well does the offline FL implementation perform compared to siloed training and the central data collection baseline?; 2) What does the trade-off between memory and performance look like for online FL? BTW, the entire thesis can be found <a href="https://www.diva-portal.org/smash/get/diva2:2083501/FULLTEXT01.pdf">here</a>.
</div>
<br>

<img src="./Images/WindTunnel.png">
<img src="./Images/Anomalies.png">

<div align="justify">
We use a causal chamber [<a href="https://www.nature.com/articles/s42256-024-00964-x">4</a>] wind tunnel device as a proxy for the actual truck component. It has actuators such as two fans and a hatch, and several pressure sensors at different positions. We can manipulate the actuators to record sensor data from a non-trivial system, where variables have first-order relations. The above figure also depicts the kind of anomalies we introduce. Point anomalies are impulses outside the normal operating range. In cumulative anomalies, a single point may be normal, but the entire window is anomalous (the signal being constant rather than sinusoidal). And lastly, contextual anomalies can be identified in the context of a small time window or with respect to another variable. So how do we generate anomalies? First, we define what normal operations are. In our case, they are sine and step signals (talking about fan loads) operating within a range of 0.5 and 1.0 with the hatch being closed. Using a combination of sine and step signals (with varying frequencies and amplitudes), we get <i>four</i> non-overlapping distributions. This is perfect for motivating FL implementation. We add anomalies by shifting the operating range between 0.1 and 0.3 for the intake and exhaust fans, opening the hatch, or adding negative impulses outside the normal range. Below, you can see the underlying data distributions.
</div>
<br>

<img src="./Images/data_stream.png">
<img src="./Images/data_stream_with_anomalies.png">
<img src="./Images/distributions.png" style='height: 100%; width: 100%; object-fit: contain'>

---

*There is not much to discuss about the experiments. You can find the hyperparameters in the thesis PDF.*   
***Obseravations***

|  |Local train 1|Local train 2|Local train 3|Local train 4|Central data collection|Offline FL|
|:-|:------------|:------------|:------------|:------------|:----------------------|:---------|
|Test 1|||||||
|Test 2|||||||
|Test 3|||||||
|Test 4|||||||
