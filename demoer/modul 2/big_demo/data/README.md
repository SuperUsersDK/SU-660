## ToDo
Kig på Volumes og Persistent Volume Claims





### Simple command

```bash
watch !! #gentag sidste kommando hvert 2. sekund

kubectl explain pods.spec.containers.env #dokumentation
kubectl api-resources #lister alle resource typer
kubectl get configmap #lister configmaps
kubectl get cm # samme som 'configmap'
kubectl config get-contexts #viser contexts

kubectl describe pod/webserver-deployment-778c65469c-7vpr9
kubectl describe pod webserver-deployment-778c65469c-7vpr9

kubectl rollout restart deployment <navn på deployment>

kubectl cordon # sæt ring om node, så der ikke lander nye pods på den
kubectl uncordon #fjern ring

kubectl drain # fjern pods gracefully fra en node (laver en cordon på noden)

kubectl get pv,pvc # Lister persistent volume, og claims

kubectl get pods -A # list alle pods i alle namespaces

Kig på affinity / anti affinity, til at tvinge pods på samme worker, og modsat

```
## Kind
```bash
kind get clusters

kind get clusters
### delete existing cluster
kind delete cluster --name clustername

### Create new cluster 
kind create cluster --name multicluster --config cluster.yaml

kubectl get nodes

#### Create namespace
kubectl create namespace demo
kubectl config set-context --current --namespace demo

#### Deploy
kubectl apply --filename deployment2.yaml

#### Existing deployment.yaml
kubectl scale deployment webserver-deployment --replicas 5


### Check pod distribution
kubectl get pods -o wide

### Get all api-resources
kubectl api-resources


### Drain pods from a node
kubectl drain multicuster-worker2 --ignore-daemonsets

```




### Noter
- Hvis man starter en pod uden refererede configmap, og så apply'er den bagefter starter pod'en af sig selv :)

- Hvis jeg nu sletter en configmap, ikke bare yaml filen, men kubernetes resources, kører pod'en stadigvæk. Men den ender i fejltilstand hvis den skal genstarte, scale eller lignende



### Your first Deployment

> name it *deployment.yaml*

```yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: webserver-deployment
spec:
  selector:
    matchLabels:
      app: webserver
  template:
    metadata:
      labels:
        app: webserver
    spec:
      containers:
        - image: nginx
          name: webserver-container
---
apiVersion: v1
kind: Service
metadata:
  name: test-service
spec:
  selector:
    app: webserver
  ports:
    - port: 80
      targetPort: 80

```

```bash

kubectl delete namespace demo01
kubectl create namespace demo01

kubectl apply --filename deployment.yaml

## Scale

kubectl scale deployment webserver-deployment --replicas 15

## View pods and deployment/replicaset

kubectl get all

### Logs all pods
kubectl logs -l app=webserver -f

## debug pod

kubectl create namespace debug && kubectl run --namespace debug debug --image nginx

### Exec into debug pod

kubectl exec -it --namespace debug debug -- bash

### DNS

[ServiceName].[Namespace]
test-service.demo01

### Port-forward
kubectl port-forward services/test-service 7777:80
```