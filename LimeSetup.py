#Lime

from lime import lime_tabular

    def limeInterpretation(self):
        """Function to setup Lime"""
        cols=['DiabTerms', 'DiabTests', 'DiabMeds', 'DiabTreatments', 'gender', 'age']
        
        #Load dataset
        df_dataset = pd.read_csv(self.dataset)
        
        #Define lime explainer
        lime_explainer = lime_tabular.LimeTabularExplainer(df_dataset[cols].values,
                                                          feature_names=cols,
                                                          class_names=['None', 'E119'])

        data = np.array(self.memberRecord)
        #data = np.array([self.f1, self.f2, self.f3, self.f4, 0, 71])
        lime_disp = lime_explainer.explain_instance(data, self.model.predict_proba)
        #lime_disp.show_in_notebook(show_table=True, show_all=False)
        lime_disp.save_to_file('ModelInterpretation.html')
