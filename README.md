## Online and Federated Learning for Predictive Maintenance

*In collaboration with Scania AB*

<div align="justify">
New heavy-duty vehicles have multiple actuators and sensors that are used generate and record time-series data streams, respectively. These data streams can be used model what normal operations in trucks are supposed to look like and by extension can also be used to identify anomalies. These anomalies could be engine overheating, malfunction in the ABS, or fatigue in suspensions. We implemented a transformer-based variational autoencoder to detect anomalies in truck operations for predictive maintenance. There are two reasons for selecting this model: 1) Because it has an explainable reconstruction loss, and 2) Because it can parallely process the entire time-series window. Now we are faced with two challenges. Firstly, we want the model to generalize well to different operating conditions. For example, if there is a truck being operated in city area and is then sent to remote hilly regions for long-haul missions, it is very likely that the underlying data distribution would change and we expect the model to perform well in either scenarios.
</div>
<br>

