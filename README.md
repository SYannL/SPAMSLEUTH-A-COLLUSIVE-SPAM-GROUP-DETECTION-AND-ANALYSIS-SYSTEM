
## SECTION 1 : PROJECT OVERVIEW
## SPAMSLEUTH: A-COLLUSIVE-SPAM-GROUP-DETECTION-AND-ANALYSIS-SYSTEM
The model used to identify spam groups in product reviews. The model consists of two parts: First, we use a locally fine-tuned large language model, llama2-7b, for preliminary spam detection and describe the fine-tuning techniques employed; second, we model the textual features of user comments and structural features of user behavior through a multi-index evaluation system, followed by spam group detection based on ranking and thresholding. 

---

## SECTION 2 : ABSTRACT
With the rise of online review platforms, the issue of fake reviews has become increasingly severe, particularly those posted by organized groups known as spam brigades, which significantly impact consumer decision-making and market fairness. To address this challenge, we have introduced a novel ranking method that combines textual content features with group behavior characteristics. Initially, we fine-tuned the large language model llama2-7b and its qlora technology to accurately assign spam scores to each reviewer's comments. Then, we designed a multi-index evaluation system for these users, and through a ranking-based method for generating candidate groups, we identified spam groups, primarily evaluating them based on user and global recall rates. We conducted extensive experiments on the YelpNYC dataset and our own collected and generated datasets, which were enhanced with real data injections to increase the realism and challenge of the tests. The experimental results showed that our method achieved a precision of 0.98 in the fine-tuned LLM's spam discriminator and a recall of 0.8 in spam group detection. Additionally, we have developed and deployed an online system that provides a practical tool to help maintain the integrity and transparency of online review platforms.



---

## SECTION 3 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

Video link: 

---

## SECTION 4 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

### [ 1 ]To run the system using iss-vm

> download pre-built virtual machine from http://bit.ly/iss-vm

> start iss-vm

> open terminal in iss-vm

> $ git clone https://github.com/SYannL/BreadcrumbsSPAMSLEUTH-A-COLLUSIVE-SPAM-GROUP-DETECTION-AND-ANALYSIS-SYSTEM

> $ source activate iss-env-py3

> (iss-env-py2) $ python run.py

### [ 2 ] To run the system in other/local machine:
### Install additional necessary libraries. This application works in python 3.6 or higher only.

> $ sudo apt-get install python-clips clips build-essential libssl-dev libffi-dev python-dev python-pip

> $ pip install pyclips flask flask-socketio eventlet simplejson pandas

> $ python run.py

> **Go to URL using web browser** http://127.0.0.1:5000/


---
