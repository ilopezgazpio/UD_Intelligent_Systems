import pandas as pd
import matplotlib.pyplot as plt

class Report:

    '''
    Class to handle all reporting and logging information for an agent
    '''


    def __init__(self):
        self.log = pd.DataFrame(columns=['Solution', 'Solution.Cost', 'Nodes.Expanded', 'Frontier.Max.Depth', 'Nodes.Frontier',])
        self.__append__(0,0,0,0,0)


    def __append__(self, solution=0, solutionCost=0, nodesExpanded=0, frontierMaxDepth=0, nodesFrontier=0):
        # Note that step iteration number is the index of the dataframe
        line = pd.Series(
            {
                'Solution' : solution,
                'Solution.Cost': solutionCost,
                'Nodes.Expanded': nodesExpanded,
                'Frontier.Max.Depth': frontierMaxDepth,
                'Nodes.Frontier': nodesFrontier
            }
        )
        self.log = self.log.append( line , ignore_index=True )


    def print_report(self):
        '''
        Prints the report of the reporting
        '''
        self.log.describe().transpose()


    def plotNumberNodesFrontier(self, show=True):
        '''
        Prints iteration vs # nodes in frontier
        '''
        ax = self.log['Nodes.Frontier'].plot(lw=2, colormap='jet', marker='.', markersize=10, title='Iteration vs # nodes in frontier')
        ax.set_xlabel("# Iteration")
        ax.set_ylabel("# Nodes")

        if show:
            plt.show()

        return ax


    def plotNodesAddedFrontier(self, nbins=20, show=True):
        '''
        Prints histogram of nodes added to frontier per step in bins
        '''

        ax = self.log['Nodes.Expanded'].plot(kind='hist', colormap='jet', bins=nbins, title='Histogram of nodes added to frontier')
        ax.set_xlabel("Nodes added")
        ax.set_ylabel("Frequency")

        if show:
            plt.show()

        return ax


    def plotFrontierMaxDepth(self, show=True):
        '''
        Print plot of Frontier Maximum Depth
        '''

        ax = self.log['Frontier.Max.Depth'].plot(lw=2, colormap='jet', marker='.', markersize=10, title='Iteration vs Frontier Maximum Depth')
        ax.set_xlabel("# Iteration")
        ax.set_ylabel("Max Depth")

        if show:
            plt.show()

        return ax


    def show(self):
        '''
        show plots
        '''
        plt.show()