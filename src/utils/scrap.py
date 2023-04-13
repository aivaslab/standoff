
testEnvs = []
for level in range(3):
    for sublevel in range(levelRanges[level]):
        tempEnv = wrap_env(para_TutorialEnv, width=9, height=9, agents=agents, step_reward=-0.2, done_reward=-1)
        tempEnv.unwrapped.random_mode = False
        tempEnv.unwrapped.mylevel = level+1
        tempEnv.unwrapped.myLevel = sublevel+1
        testEnvs += [VecMonitor(pz2sb3(tempEnv), )]


envInstance.reset()
video_length = 101
random_policy = True
max_vids = 3
cur_vids = 0


env = from_parallel(p_env.unwrapped.vec_envs[0].par_env).unwrapped
env.random_mode = False

while True:
    obs = env.reset()
    pos = next(iter(env.agent_instances)).pos
    #print(pos)
    break
ims = [env.render()]

rew_success = False
key_success = True

print('sl',env.mysublevel)

for i in range(video_length+1):
    show_state(env, step=i)
    if random_policy:
        actions = { a: env.action_spaces[a].sample() for k, a in enumerate(env.possible_agents) }
    else:
        actions = {x: model.predict(obs[x])[0] for x in env.possible_agents}
    obs, rew, dones, _, = env.step(actions)
    
env.close()