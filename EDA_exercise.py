import streamlit as st

import os
import pandas as pd
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Common ML Data Explorer"""
    st.title("Common ML Dataset Explorer")
    st.subheader("Simple DataScience App with Streamlit")

    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:60px;"> Streamlit is Awesome</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    def file_selector(folder_path='datasets'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.write("You selected '%s''" % filename)
    df = pd.read_csv(filename)

    if st.checkbox("Show DataSet"):
        num = st.number_input("Number of Rows to View", value=1)
        st.dataframe(df.head(num))

    if st.button("Columns Names"):
        st.write(df.columns)

    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension by",("Rows","Columns"))
        if data_dim == 'Rows':
            st.text("Number of Rows")
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])

    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect('Select',all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    if st.button("Data Types"):
        st.write(df.dtypes)

    if st.button("Value Counts"):
        st.text("Value Counts By Target/Class")
        st.write(df.iloc[:,-1].value_counts())

    if st.checkbox("Summary"):
        st.write(df.describe())

    st.subheader("Data Visualization")

    if st.checkbox("Correlation Plot [Matplotlib]"):
        plt.matshow(df.corr())
        st.pyplot

    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target/Class")

        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox('Select Primary Column To Group By',all_columns_names)
        selected_columns_names = st.multiselect('Select Columns',all_columns_names)
        if st.button("Plot"):
            st.text("Generating Plot for: {} and {}".format(primary_col,selected_columns_names))
            if selected_columns_names:
                vc_plot = df.groupby(primary_col)[selected_columns_names].count()
            else:
                vc_plot = df.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()

    if st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()

        if st.button("Generate Pie Plot"):
            st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    if st.checkbox("BarH Plot"):
        all_columns_names = df.columns.tolist()
        st.info("Please Choose the X and Y Column")
        x_column = st.selectbox('Select X columns For Barh Plot',all_columns_names)
        y_column = st.selectbox('Select Y Columns For Varh Plot',all_columns_names)
        barh_plot = df.plot.barh(x=x_column,y=y_column,figsize=(10,10))
        if st. button("Generate Barh Plot"):
            st.write(barh_plot)
            st.pyplot()

    st.subheader("Customizable Plots")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select the Type of Plot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect('Select Columns To Plot',all_columns_names)
    cust_target = df.iloc[:,-1].name

    if st.button("Generate Plot"):
        st.success("Generating A Customizable Plot of: {} for :: {}".format(type_of_plot,selected_columns_names))

        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)
        elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)
        elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)
        elif type_of_plot == 'hist':
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot,bins=2)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'box':
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'kde':
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        else:
            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()



    st.subheader("Our Feature and Target")

    if st.checkbox("Show Features"):
        all_features = df.iloc[:,0:-1]
        st.text('Features Names:: {}'.format(all_features.columns[0:-1]))
        st.dataframe(all_features.head(10))

    if st.checkbox("Show Target"):
        all_target = df.iloc[:,-1]
        st.text('Target/Class Name:: {}'.format(all_target.name))
        st.dataframe(all_target.head(10))

        st.markdown("""[google.com](iris.zip""")
        st.markdown("""[google.com](./iris.zip)""")

        def makezipfile(data):
            output_filename = '{}_zipped.zip'.format(data)
            with ZipFile(output_filename,"W")as z:
                z.write(data)
            return output_filename
        if st.button("Download File"):
            DOWNLOAD_TPL = f'[{filename}]({makezipfile(filename)})'

            st.text(DOWNLOAD_TPL)
            st.markdown(DOWNLOAD_TPL)




if __name__ == '__main__':
    main()