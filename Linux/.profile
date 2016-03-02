# ~/.profile

### SET ARCHITECTIRE FLAGS
export ARCHFLAGS="-arch x86_64"


### PIP SETUP
# pip should only run if there is a virtualenv currently activated
export PIP_REQUIRE_VIRTUALENV=“”
# cache pip-installed packages to avoid re-downloading
export PIP_DOWNLOAD_CACHE=$HOME/.pip/cache

syspip(){
PIP_REQUIRE_VIRTUALENV="" pip "$@"
}


### SET ENV VARS
export ORACLE_HOME=/Users/amirkavousian/oracle/instantclient_11_2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ORACLE_HOME
export DYLD_LIBRARY_PATH=$ORACLE_HOME
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/mysql/lib/  # for perl and dbd mysql
export VERSIONER_PYTHON_PREFER_32_BIT=yes
export PATH=/usr/local/bin:/usr/local/share/python:/Library/Frameworks/Python.framework/Versions/3.4/bin:/usr/local/bin:/Users/amirkavousian/anaconda/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/git/bin:/Users/amirkavousian/oracle/instantclient_11_2:/usr/local/mysql/bin
#export PATH=$PATH:/Users/amirkavousian/oracle/instantclient_11_2
#export PATH=/usr/local/bin:/usr/local/share/python:$PATH
export OCI_LIB=$ORACLE_HOME
# for RJDBC
export JAVA_HOME=/usr/libexec/java_home
# for Maven
#export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.7.0_72.jdk/Contents/Home
export rdev_gits=/Users/amirkavousian/gits/performance-engineering
#export PYTHONPATH=/usr/local/lib/python:/usr/local/bin/python:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME:/usr/local/lib/python:/usr/local/bin/python:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH
export proj_dir=/Users/amirkavousian/Documents/PROJECTS
export pytut=/Users/amirkavousian/Documents/Py_Codes

# For MySQL cookbook. Change later if installing a new program that depends on this
MYSQL_HOME=/usr/local/mysql

# Setting PATH for Python 3.4. The orginal version is saved in .bash_profile.pysave
#export PATH="/Library/Frameworks/Python.framework/Versions/3.4/bin:${PATH}"

# uncomment if you want anaconda python to be the default python
# export PATH="/Users/amirkavousian/anaconda/bin:$PATH"


### BASH COMPLETION TOOL
if [ -f $(brew --prefix)/etc/bash_completion ]; then
. $(brew --prefix)/etc/bash_completion
fi


### GIT COMPLETION
source /Users/amirkavousian/git-completion.bash


### COLOR CODE TERMINAL
function parse_git_branch {
git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

function proml {
local BLUE="\[\033[0;34m\]"
local RED="\[\033[0;31m\]"
local LIGHT_RED="\[\033[1;31m\]"
local GREEN="\[\033[0;32m\]"
local LIGHT_GREEN="\[\033[1;32m\]"
local WHITE="\[\033[1;37m\]"
local LIGHT_GRAY="\[\033[0;37m\]"
case $TERM in
xterm*)
TITLEBAR='\[\033]0;\u@\h:\w\007\]'
;;
*)
TITLEBAR=""
;;
esac

PS1="${TITLEBAR}\
$GREEN[$GREEN\u@\h:\w$LIGHT_GRAY\$(parse_git_branch)$GREEN]\
$GREEN\$ "
PS2='> '
PS4='+ '
}
proml

### For Apache Spark
if which java > /dev/null; then export JAVA_HOME=$(/usr/libexec/java_home); fi

# For a ipython notebook and pyspark integration
if which pyspark > /dev/null; then
  export SPARK_HOME="/usr/local/Cellar/apache-spark/1.4.1/libexec/"
  export PYSPARK_SUBMIT_ARGS="--master local[2]"
  export SCALA_HOME="/usr/local/bin/scala"
fi

# to open PyCharm with env vars available
alias pycharmopen='/Applications/PyCharm\ CE.app/Contents/MacOS/pycharm ; exit;'
alias ssh_ec2='ssh -i ~/Documents/AWS/sr-rdev-oregon.pem ubuntu@ec2-52-12-32-148.us-west-2.compute.amazonaws.com'

