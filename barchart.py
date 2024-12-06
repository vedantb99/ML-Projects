#https://www.youtube.com/watch?v=mafzIn8TneQ
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anime

def setup_plot_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_xlabel('Total Population on 1st January')
    ax.set_title('Top 10 Countries by Population(in milions)')
def add_year_text(ax, year):
    ax.text(0.9,0.1,str(year), transform = ax.transAxes, ha='center', fontsize=20)
def create_animation(df):
    frames = df['Time'].unique()

    fig,axs = plt.subplots(figsize=(12,6))
    


    def animate(frame):
        axs.clear()
        pop_data_frame = df[df['Time'] == frame]
        top_countries = pop_data_frame.nlargest(10,'TPopulation1Jan').sort_values('TPopulation1Jan', ascending=True)
        axs.barh(top_countries['Location'], top_countries['TPopulation1Jan'])
        for i,row in top_countries.iterrows():
            axs.text(row['TPopulation1Jan'],row['Location'], f'{row["TPopulation1Jan"]:,.2f}', va= "center")
        
        setup_plot_style(axs)
        add_year_text(axs,frame)
        plt.tight_layout()
    a = anime.FuncAnimation(fig,animate, frames = frames, interval = 200)
    return a
if __name__ == '__main__':
    df = pd.read_csv('./data/cleaned-data.csv')
    a = create_animation(df)
    # a.save('barChartAnimation.gif', writer='pillow', fps=20)
    plt.show() 