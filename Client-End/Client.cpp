//
// Created by Angel on 15/9/2023.
//

#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        std::cerr << "Could not create socket." << std::endl;
        return -1;
    }

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(65432);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed." << std::endl;
        close(sock);
        return -1;
    }

    // Send 3 pieces of data
    std::string edge_attribute = "[915,1830] [5694,11323] [5694,65] [380,760] [1015,2030] [1035,0] [5621,11242] [2545,5090] [401,802] [404,412] [404,396] [5264,10528] [1062,2124] ";
    std::string edge_index = "[0,1] [2,3] [3,2] [5,4] [1,2] [7,9] [2,5] [10,5] [3,6] [5,3] [3,5] [4,8] [11,8] ";
    std::string graph_data = "[2,-1] [2,-1] [3,-1] [3,-1] [2,-1] [4,-1] [2,-1] [3,0] [3,-1] [1,1] [1,-1] [1,-1] ";
    std::string tabular_data = "3,0.5,73,15,1035,2.48697,2.49449,2.49935,2.50327,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0,0,2,0,0,0,0,0,0,1 ";


    std::string combined_data = edge_attribute + "|" + edge_index + "|" + graph_data + "|" + tabular_data + "|";

    std::cout << "Received from server:\n" << combined_data << std::endl;


    send(sock, combined_data.c_str(), combined_data.size(), 0);


    // Receive the processed result
    char buffer[1024] = {0};
    recv(sock, buffer, sizeof(buffer), 0);
    std::cout << "Received from server: " << buffer << std::endl;

    close(sock);
    return 0;
}
