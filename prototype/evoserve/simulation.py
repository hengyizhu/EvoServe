"""CPU cameras on a separate simulator state; physics never waits on rendering."""
import queue
import threading
import time

class ObservationRenderer:
    def __init__(self, env):
        self.env=env
        self.cameras=[v for v in env.env._observables.values() if v.modality=='image']
        if not self.cameras:raise RuntimeError('No LIBERO image observables found')
        self.disable()
    def disable(self):
        for v in self.cameras:v.set_enabled(False)
    def observation(self):
        for v in self.cameras:v.set_enabled(True)
        try:return self.env.env._get_observations(force_update=True)
        finally:self.disable()

class AsyncRenderer:
    """Single independent OSMesa context, latest-state mailbox, tagged frames."""
    def __init__(self, bddl, scene=None):
        self.scene=scene
        self.bddl=bddl;self.jobs=queue.Queue(maxsize=1);self.latest=None;self.error=None
        self.stop=threading.Event();self.lock=threading.Lock()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def submit(self, episode, frame, state):
        item=(episode,frame,state,time.monotonic())
        try:self.jobs.put_nowait(item)
        except queue.Full:
            try:self.jobs.get_nowait()
            except queue.Empty:pass
            try:self.jobs.put_nowait(item)
            except queue.Full:pass
    def get(self,episode):
        if self.error:raise RuntimeError(self.error)
        with self.lock:
            return self.latest if self.latest and self.latest['episode']==episode else None
    def run(self):
        env=None
        try:
            from libero.libero.envs import OffScreenRenderEnv
            env=OffScreenRenderEnv(bddl_file_name=self.bddl,camera_heights=256,camera_widths=256,hard_reset=False)
            env.reset()
            if self.scene and self.scene['kind'] in ('camera_orbit','table_texture'):
                from scene_perturb import apply_scene
                apply_scene(env,self.scene)
            renderer=ObservationRenderer(env)
            while not self.stop.is_set():
                try:episode,frame,state,captured=self.jobs.get(timeout=.1)
                except queue.Empty:continue
                env.set_state(state);env.sim.forward();env._post_process()
                obs=renderer.observation()
                with self.lock:self.latest=dict(episode=episode,frame=frame,obs=obs,captured=captured,rendered=time.monotonic())
        except Exception:
            import traceback
            self.error=traceback.format_exc()
        finally:
            if env:env.close()
    def close(self):self.stop.set();self.thread.join(timeout=10)
