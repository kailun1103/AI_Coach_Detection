# ---Buiid Env
python -m venv venv 

venv\Scripts\activate # (PS) activate env
deactivate #exit Env.

clear ; git add . ; git commit -m "20241217 Feedback function DONE" ; git push -u origin max-qat

clear ; git checkout max-prod ; git merge max-qat ; git add . ; git commit -m "20241213" ; git push -u origin max-prod

clear ; git checkout max-qat







# ---假設要將qat合併到main，確認當前qat已經是最新提交
git checkout main
git merge qat

#---Push GIT
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/MaxLu2002/Resume_Chia-Yang-Lu.git

git push -u origin main # 可以加 --force 覆蓋遠端
git push origin A:B # 遠端分支 B 會被創建或更新，內容來自本地分支A。


#---GIT Branch指令
git branch <new branch>
git switch <target branch>
git branch -d <branch name> (d= delete , must leave the branch)
git checkout -b max-prod
git branch -vv 查看當前本地分支關聯到哪個遠端分支
git branch --unset-upstream 執行以下指令即可取消某個本地分支與遠端分支的關聯：



#---創建或導航到目標目錄
mkdir -p ~/my_project
cd ~/my_project