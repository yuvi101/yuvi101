---
title: "Are You a Hard Worker Or Just Lucky?"
date: 2021-09-09T09:43:47+03:00
#draft: true
author: "Yuval"

tags: ["Python"]
categories: ["Beginners"]
---
Have you ever wondered why are some people more successful than others?
Is it they’ve got a better education, more connections? Are they more hard-working? Or maybe they are just luckier!?
<!--more-->

{{< admonition type=note title="Note" open=false >}}
The BBC series "Astronauts: Do You Have What It Takes?" was the inspiration for writing this post.
{{< /admonition >}}

![Astronaut on the moon >](https://images.unsplash.com/photo-1541873676-a18131494184?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=418&q=80)

{{< admonition type=info title="Info" open=true >}}
In 2017, NASA received **18,300** applications and selected just **12** candidates who passed to the next level.
{{< /admonition >}}

To test whether luck played a major role in the selection of contestants we are going to run a model of the selection process. But in order to do so, first, we will assume a few things:
* Success consists of 90% skills/education/connections etc. and only 10% of luck.
* Assume that the selection process contains only 1 stage, meaning the 12 candidates are chosen out of the 18,300 applicants, without any intermediate stages.

I generated for each of the 18,300 candidates a skill score and a luck score between 0 and 100. Then calculated overall score with the previously mentioned scores weighted, 90% skill and 10% luck. Our goal is to choose the top 12 candidates, and then check if luck played any role in the selection process.

```Python
import random

candidates = []
for k in range(100):
    for i in range(18300):
        #generating random numbers for skill and luck
        skill = random.random()
        luck = random.random()
        #candidates.append(overall score, skill, luck, id)
        candidates.append([round(skill*0.9 + luck*0.1, 4),round(skill*0.9, 4) , round(luck*0.1, 4), str(i)])
```

In the next step, I made different lists for the 12 selected upon overall score and skill only.
I then compared the 2 lists, in order to see how many of the candidates appear in both lists, based on their IDs.

```Python
candidates = sorted(candidates)
#top 12 candidates based on overall scores
chosen_twelve = candidates[-12:]
#sorting candidates list by their skill score and assigning them to a #new list
candidates.sort(key=lambda x:x[1])
top12_skilled = candidates[-12:]
weighted_skill_scores = []
weighted_luck_scores = []
ids = []
ids_top12_skilled = []
for j in range(12):
    ids_top12_skilled.append(top12_skilled[j][3])
sum = sum + (len([id for id in ids if id in ids_top12_skilled]))
avg = sum/100
print(avg)
```

In order to minimize the chances of coincidence, the whole program runs 100 times, after which we calculate the average number of people that are in both groups, in the "top12_skilled" and in the "chosen_twelve".  
The 'avg' variable represents the average number of people who would be accepted to the space program, with or without the luck scores.
The result was ~ 1.8, in other words only 1 or maybe 2 out of the 12 chosen, would have been chosen even if they were completely unlucky.

We can now conclude that luck does play a pretty significant role in our lives, even if it constitutes only 10 percent of our achievements.

So if you are doing pretty well, be aware that there is a drop of luck in this,
but if you encountering some difficulties right now, don't be hard with yourself, keep working hard, and luck will come.


p.s feel free to make changes in this code and run it on your machine :)
