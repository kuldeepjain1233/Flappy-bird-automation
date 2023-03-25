import pygame
import neat
import time
import os
import random

pygame.font.init()

Gen = 0
win_width = 600
win_height = 800

bird_imgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]
pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "base.png")))
bg_img = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "background.png")))
stat_font = pygame.font.SysFont("comicsans", 50)

def main():
    class Bird:
        imgs = bird_imgs
        max_rotation = 25
        rot_vel = 20
        animation_time = 5

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.tilt = 0
            self.tick_count = 0
            self.vel = 0
            self.height = self.y
            self.img_count = 0
            self.img = self.imgs[0]

        def jump(self):
            self.vel = -10.5
            self.tick_count = 0
            self.height = self.y

        def move(self):
            self.tick_count += 1

            d = self.vel*self.tick_count + 1.5*self.tick_count**2

            if d >= 16:
                d = 16
            if d < 0:
                d -= 2

            self.y = self.y + d
            if d < 0 or self.y < self.height + 50:
                self.tilt = self.max_rotation
            else: 
                if self.tilt > 90:
                    self.tilt -= self.rot_vel

        def draw(self, win):
            self.img_count += 1

            if self.img_count < self.animation_time:
                self.img = self.imgs[0]
            elif self.img_count < self.animation_time*2:
                self.img = self.imgs[1]
            elif self.img_count < self.animation_time*3:
                self.img = self.imgs[2]
            elif self.img_count < self.animation_time*4:
                self.img = self.imgs[1]
            elif self.img_count == self.animation_time*4 + 1:
                self.img = self.imgs[0]
                self.img_count = 0

            if self.tilt <= -80:
                self.img = self.imgs[1]
                self.img_count = self.animation_time*2
            
            rotated_image = pygame.transform.rotate(self.img, self.tilt)
            new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
            win.blit(rotated_image, new_rect.topleft)

        def get_mask(self):
            return pygame.mask.from_surface(self.img)

    class Pipes:
        gap = 200
        vel = 100

        def __init__(self, x):
            self.x = x
            self.height = 0

            self.top = 0
            self.bottom = 0
            
            self.pipe_top = pygame.transform.flip(pipe_img, False, True)
            self.pipe_bottom = pipe_img

            self.passed = False
            self.set_height()

        def set_height(self):
            self.height = random.randrange(50, 450)
            self.top = self.height - self.pipe_top.get_height()
            self.bottom = self.height + self.gap

        def move(self):
            self.x -= self.vel

        def draw(self, win):
            win.blit(self.pipe_top, (self.x, self.top))
            win.blit(self.pipe_bottom, (self.x, self.bottom))

        def collide(self, bird):
            bird_mask = bird.get_mask()
            top_mask = pygame.mask.from_surface(self.pipe_top)
            bottom_mask = pygame.mask.from_surface(self.pipe_bottom)

            top_offset = (self.x - bird.x, self.top - round(bird.y))
            bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

            b_point = bird_mask.overlap(bottom_mask, bottom_offset)
            t_point = bird_mask.overlap(top_mask, top_offset)

            if t_point or b_point:
                return True

            return False

    class Base:
        vel = 5
        width = base_img.get_width()
        img = base_img

        def __init__(self, y):
            self.y = y
            self.x1 = 0 
            self.x2 = self.width

        def move(self):
            self.x1 -= self.vel
            self.x2 -= self.vel

            if self.x1 + self.width < 0:
                self.x1 = self.x2 + self.width

            if self.x2 + self.width < 0:
                self.x2 = self.x1 + self.width
        
        def draw(self, win):
            win.blit(self.img, (self.x1, self.y))
            win.blit(self.img, (self.x2, self.y))

    def draw_window(win, birds, pipes, base, score, gen):
        win.blit(bg_img, (0,0))
        
        for pipe in pipes:
            pipe.draw(win)

        text = stat_font.render("score: " + str(score) ,1, (255,255,255))
        win.blit(text, (win_width - 10 - text.get_width(), 10))

        text = stat_font.render("Gen: " + str(gen) ,1, (255,255,255))
        win.blit(text, (10,10))

        base.draw(win)
        
        for bird in birds:
            bird.draw(win)

        pygame.display.update()

    def main(genomes, config):
        global Gen
        Gen += 1
        nets = []
        ge = []
        birds = []

        for _, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(230, 350))
            ge.append(genome)

        base = Base(730)
        pipes = [Pipes(700)]
        win = pygame.display.set_mode((win_width, win_height))
        clock = pygame.time.Clock()
        score = 0
        run = True

        while run and len(birds) > 0 :
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.QUIT()

                    quit()
            
            pipe_ind = 0
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                    pipe_ind = 1

            for x, bird in enumerate(birds):
                ge[x].fitness += 0.1
                bird.move()

                output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

                if output[0] > 0.5:
                    bird.jump()
            base.move()

            add_pipe = False
            rem = []
            for pipe in pipes:
                for bird in birds:
                    if pipe.collide(bird):
                        #ge[x].fitness -= 1
                        nets.pop(birds.index(bird))
                        ge.pop(birds.index(bird))
                        birds.pop(birds.index(bird))
                
                    if not pipe.passed and pipe.x < bird.x:
                        pipe.passed = True
                        add_pipe = True
                
                if pipe.x + pipe.pipe_top.get_width() < 0:
                    rem.append(pipe)
                
                pipe.move()

            if add_pipe:
                score += 1
                for g in ge:
                    g.fitness += 5
                pipes.append(Pipes(700))

            for r in rem:
                pipes.remove(r)


            for bird in birds:
                if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            bird.move()
            draw_window(win, birds, pipes, base, score, Gen)

    def run(config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        p.run(main,50)

    if __name__ == '__main__':
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config.txt')
        run(config_path)

main()