OLD_CSUM=`echo $PATH | grep -oP “(?<=\/home\/$USER\/.vscode-server\/bin\/).*?(?=\/bin)” | head -1`
NEW_CSUM=`ls -tr /home/$USER/.vscode-server/bin/ | tail -n 1`export PATH=`echo $PATH | sed “s/$OLD_CSUM/$NEW_CSUM/g”`
export GIT_ASKPASS=`echo $GIT_ASKPASS | sed “s/$OLD_CSUM/$NEW_CSUM/g”`
export VSCODE_GIT_ASKPASS_MAIN=`echo $VSCODE_GIT_ASKPASS_MAIN | sed “s/$OLD_CSUM/$NEW_CSUM/g”`
export VSCODE_GIT_ASKPASS_NODE=`echo $VSCODE_GIT_ASKPASS_NODE | sed “s/$OLD_CSUM/$NEW_CSUM/g”`
export VSCODE_IPC_HOOK_CLI=`ls -tr /tmp/vscode-ipc-* | tail -n 1`%

